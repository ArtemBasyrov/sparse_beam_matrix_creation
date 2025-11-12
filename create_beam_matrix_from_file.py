import numpy as np
import healpy as hp
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R
import argparse
from pixell import enmap
from numba import jit
import numba as nb
import pandas as pd
import os
import time
from datetime import timedelta


def read_beam_file(path_to_file, field=0):
    beam_map = enmap.read_map(path_to_file)
    pixel_map = np.array(beam_map[field]).copy()

    ra, dec = beam_map.posmap()
    ra_c = ra[100,100]
    dec_c = dec[100,100]
    ra = np.array(ra-ra_c).copy()
    dec = np.array(dec-dec_c).copy()

    return pixel_map, ra, dec


@jit(nopython=True)
def fast_intersect1d_numba(arr1, arr2):
    """Numba-optimized intersection that actually works"""
    # Create a set-like structure for faster lookups
    max_val = max(np.max(arr1) if len(arr1) > 0 else 0, 
                  np.max(arr2) if len(arr2) > 0 else 0)
    
    # Use boolean array for membership testing
    if max_val < 1000000:  # Only use this method for reasonable sizes
        lookup = np.zeros(int(max_val) + 1, dtype=nb.boolean)
        for val in arr2:
            if val <= max_val:
                lookup[val] = True
        
        result = []
        for val in arr1:
            if val <= max_val and lookup[val]:
                result.append(val)
        return np.array(result)
    else:
        # Fallback to simple method for large ranges
        result = []
        for val in arr1:
            for val2 in arr2:
                if val == val2:
                    result.append(val)
                    break
        return np.array(result)


def beam_stacking_optimized(pixel_map_sel, pix_patch, ipix_neighbours):
    """Using pandas for efficient groupby with NaN handling"""
    # Create DataFrame for groupby operations
    df = pd.DataFrame({
        'pixel': pix_patch,
        'value': pixel_map_sel
    })
    
    # Group by pixel and compute mean, automatically ignoring NaNs
    grouped = df.groupby('pixel')['value'].mean()
    
    # Extract values for our neighbours
    beam_stacked = grouped.reindex(ipix_neighbours).fillna(0).values
    
    # Normalize
    total = np.sum(beam_stacked)
    if np.abs(total) > 0:
        beam_stacked /= total
    
    return beam_stacked


def recenter_coordinates_vector_from_rotation(ra, dec, rot_vec, sel=None):
    """Vectorized version for batch processing"""
    def spherical_to_vector(phi, theta):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack([x, y, z], axis=-1)
    
    # Convert input coordinates to vectors
    phi_orig = ra
    theta_orig = np.pi/2 - dec
    vec_orig = spherical_to_vector(phi_orig, theta_orig)

    # Apply selection mask if provided
    if sel is not None:
        vec_orig_sel = vec_orig[sel].reshape(-1, 3)
    else:
        vec_orig_sel = vec_orig.reshape(-1, 3)
    
    rot = R.from_rotvec(rot_vec)
    vec_rotated = rot.apply(vec_orig_sel)
    
    return vec_rotated


def precompute_rotation_vector(ra, dec, phi_pix, theta_pix, center_idx=(100, 100), save_path=None):
    def spherical_to_vector(phi, theta):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack([x, y, z], axis=-1)
    
    phi_orig = ra
    theta_orig = np.pi/2 - dec

    vec_center = spherical_to_vector(phi_orig[center_idx], theta_orig[center_idx])
    vec_target = spherical_to_vector(phi_pix, theta_pix)
    
    # Ensure vec_center has the right shape for broadcasting
    vec_center = vec_center[np.newaxis, :]  # Shape: (1, 3)
    
    # Compute cross product and normalize
    axis = np.cross(vec_center, vec_target)  # vec_center broadcasts to match vec_target
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    axis = axis / axis_norm
    
    # Compute dot product along the last axis
    dot_product = np.sum(vec_center * vec_target, axis=-1)
    angle = np.arccos(np.clip(dot_product, -1, 1))
    
    rot_vector = axis * angle[..., np.newaxis]
    
    # Save to file if path is provided
    if save_path is not None:
        np.save(save_path, rot_vector)
        print(f"Saved rotation vectors to {save_path}")
    
    return rot_vector


def beam_from_file(nside, path_to_file, field=0, observed_pixels=None):
    npix = hp.nside2npix(nside)

    if observed_pixels is None:
        observed_pixels = np.arange(npix)

    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(npix))

    data = []
    row = []
    col = []

    pixel_map, ra, dec = read_beam_file(path_to_file, field=field)
    
    # "We have checked the contribution to cosmological analysis of the beam as 
    # a function of angular scale and basically anything that happens under -25 dB 
    # doesn't make any differnce to sigma(r) or other parameters uncertainty"
    # Nadia Dachlythra
    sel = (10*np.log10(np.abs(pixel_map)) > -25) # K_RJ -> dB
    pixel_map_sel = pixel_map[sel]
    
    # Check if precomputed rotation vectors exist
    rotation_file = f"rotation_vectors_nside_{nside}.npy"
    if os.path.exists(rotation_file):
        print(f"Loading precomputed rotation vectors from {rotation_file}")
        rot_vectors = np.load(rotation_file)
    else:
        print("Precomputing rotation vectors...")
        rot_vectors = precompute_rotation_vector(
            ra, dec, phi_pix, theta_pix, 
            center_idx=(100, 100), save_path=rotation_file
        )
    
    # Initialize timing variables
    total_rotation_time = 0
    total_stacking_time = 0
    
    n_observed = len(observed_pixels)
    print(f"Starting main processing loop for {n_observed} pixels...")
    loop_start_time = time.time()
    last_report_time = loop_start_time
    
    for i, pix_idx in enumerate(observed_pixels):
        # Time rotation operation
        rot_start = time.time()
        rot_vec = rot_vectors[pix_idx]
        vec = recenter_coordinates_vector_from_rotation(ra, dec, rot_vec, sel=sel)
        total_rotation_time += time.time() - rot_start
        
        try:
            pix_patch = hp.vec2pix(nside, x=vec[:,0], y=vec[:,1], z=vec[:,2])
        except ValueError as e:
            print(f"Pixel index {pix_idx} (observed index {i}) caused an error in vec2pix: {e}")
            print(f"vec: {vec}")
            continue
        
        ipix_neighbours = np.unique(pix_patch)
        if len(ipix_neighbours) == 0:
            continue

        # Time stacking operation
        stacking_start = time.time()
        beam_stacked = beam_stacking_optimized(pixel_map_sel, pix_patch, ipix_neighbours)
        total_stacking_time += time.time() - stacking_start

        # Only add if we have valid beam values
        if len(beam_stacked) > 0:
            data.extend(beam_stacked)
            row.extend([i] * len(ipix_neighbours))
            col.extend(ipix_neighbours)

        if i % 100000 == 0:
            current_time = time.time()
            elapsed_since_last = current_time - last_report_time
            total_elapsed = current_time - loop_start_time
            pixels_per_sec = 100000 / elapsed_since_last if elapsed_since_last > 0 else 0
            estimated_total = (total_elapsed / (i + 1)) * n_observed if i > 0 else 0
            remaining = estimated_total - total_elapsed
            
            print(f"Processed pixel {i}/{n_observed} "
                  f"({i/n_observed*100:.1f}%) - "
                  f"Rate: {pixels_per_sec:.1f} pix/sec - "
                  f"ETA: {timedelta(seconds=int(remaining))}")
            last_report_time = current_time

    loop_end_time = time.time()
    loop_total_time = loop_end_time - loop_start_time
    
    # Print detailed timing information
    print(f"\nMain loop timing breakdown for {n_observed} pixels:")
    print(f"Total loop time: {loop_total_time:.2f} seconds")
    print(f"Average time per pixel: {loop_total_time/n_observed*1000:.2f} ms")
    print(f"  Rotation: {total_rotation_time:.2f}s ({total_rotation_time/loop_total_time*100:.1f}%)")
    print(f"  Stacking: {total_stacking_time:.2f}s ({total_stacking_time/loop_total_time*100:.1f}%)")
    print(f"Processing rate: {n_observed/loop_total_time:.2f} pixels/second")
    
    sparse_beam_ij_query = sp.csr_array((data, (row, col)), shape=(n_observed, npix))
    return sparse_beam_ij_query


def pad_to_max_neighbors(B):
    """Pad all rows to have the same number of non-zero elements as the maximum row.
    
    This preserves all beam information while making the matrix uniform for efficient processing.
    """
    if not sp.isspmatrix_csr(B):
        B = B.tocsr()
    
    n_rows = B.shape[0]
    
    # Find maximum number of non-zero elements in any row
    nnz_per_row = np.diff(B.indptr)
    max_nnz = np.max(nnz_per_row)
    
    # Create padded arrays
    padded_data = np.zeros((n_rows, max_nnz), dtype=B.dtype)
    padded_indices = np.zeros((n_rows, max_nnz), dtype=np.int32)
    
    for i in range(n_rows):
        start = B.indptr[i]
        end = B.indptr[i+1]
        row_size = end - start
        
        if row_size > 0:
            padded_data[i, :row_size] = B.data[start:end]
            padded_indices[i, :row_size] = B.indices[start:end]
    
    return padded_data, padded_indices, max_nnz


def save_beam_matrix(filename, data_array, indices_array, observed_pixels, nside, max_nnz):
    """Save beam matrix in compressed format."""
    np.savez_compressed(filename,
             data=data_array,
             indices=indices_array,
             observed_pixels=observed_pixels,
             nside=nside,
             max_nnz=max_nnz)
    
    print(f"Saved sparse beam matrix to {filename} with shape {data_array.shape}.")


def process_single_field(nside, path_to_file, field, observed_pixels):
    """Process a single field and save its beam matrix"""
    print(f"\n{'='*60}")
    print(f"Processing field {field}...")
    print(f"{'='*60}")
    
    B_ij = beam_from_file(nside, path_to_file, field=field, observed_pixels=observed_pixels)
    print(f"Beam matrix after creation shape: {B_ij.shape}, max number of neighbors per pixel: {np.max(np.diff(B_ij.indptr))}")

    # eliminate very small values to save space - using absolute value
    B_ij.data[np.abs(B_ij.data) < 1e-15] = 0 # np.float64 precision
    B_ij.eliminate_zeros()
    print(f"Beam matrix after eliminating small values shape: {B_ij.shape}, max number of neighbors per pixel: {np.max(np.diff(B_ij.indptr))}")

    # equalize the number of non-zero elements in each row
    # Pad to maximum number of neighbors instead of cutting to minimum
    data_array, indices_array, max_nnz = pad_to_max_neighbors(B_ij)
    print(f"Max neighbors per pixel: {max_nnz}")

    field_dict = {0: 'I', 1: 'Q', 2: 'U'}
    field_label = field_dict.get(field, 'I')
    filename = "beam_sparse_{0}_cutoff_satp1_ws0_{1}.npz".format(nside, field_label)
    save_beam_matrix(filename, data_array, indices_array, observed_pixels, nside, max_nnz)
    
    return filename


def parse_field_argument(field_arg):
    """Parse field argument which can be numbers (0,1,2) or letters (I,Q,U)"""
    field_mapping = {
        '0': 0, '1': 1, '2': 2,
        'I': 0, 'Q': 1, 'U': 2,
        'i': 0, 'q': 1, 'u': 2
    }
    
    if isinstance(field_arg, list):
        fields = []
        for f in field_arg:
            if isinstance(f, int):
                if f in [0, 1, 2]:
                    fields.append(f)
                else:
                    raise ValueError(f"Invalid field value: {f}. Must be 0,1,2 or I, Q, U")
            else:
                f_str = str(f).strip()
                if f_str in field_mapping:
                    fields.append(field_mapping[f_str])
                else:
                    raise ValueError(f"Invalid field value: {f}. Must be 0,1,2 or I,Q,U or i,q,u")
        return fields
    else:
        # Single value
        if isinstance(field_arg, int):
            if field_arg in [0, 1, 2]:
                return [field_arg]
            else:
                raise ValueError(f"Invalid field value: {field_arg}. Must be 0,1,2 or I, Q, U")
        else:
            field_str = str(field_arg).strip()
            if field_str in field_mapping:
                return [field_mapping[field_str]]
            else:
                raise ValueError(f"Invalid field value: {field_arg}. Must be 0,1,2 or I,Q,U or i,q,u")


def main(nside, path_to_file, fields, observed_pixels_file=None):
    """Process multiple fields"""
    # Load observed pixels if provided
    if observed_pixels_file:
        observed_pixels = np.load(observed_pixels_file)

        # check that all elements are unique in observed_pixels
        unique_pixels = np.unique(observed_pixels)
        if len(unique_pixels) != len(observed_pixels):
            raise ValueError("Observed pixels array contains duplicate entries.")
        
        print(f"Loaded {len(observed_pixels)} observed pixels from {observed_pixels_file}")
    else:
        observed_pixels = None
        print("Using full sky coverage")
    
    saved_files = []
    for field in fields:
        filename = process_single_field(nside, path_to_file, field, observed_pixels)
        saved_files.append(filename)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create beam matrix for given nside.")
    parser.add_argument("--nside", type=int, help="HEALPix nside parameter.")
    parser.add_argument("--file", type=str, help="Path to pixel map FITS file.")
    parser.add_argument("--field", nargs='+', default=['0'], 
                       help="Field index in the FITS file. Can be: 0,1,2 or I,Q,U or i,q,u. Multiple values can be provided. (default: 0)")
    parser.add_argument("--observed_pixels", type=str, help="Optional: .npy file with observed pixel indices")

    args = parser.parse_args()
    nside = args.nside
    path_to_file = args.file
    observed_pixels_file = args.observed_pixels
    
    # Parse field argument(s)
    try:
        fields = parse_field_argument(args.field)
    except ValueError as e:
        print(f"Error parsing field argument: {e}")
        exit(1)

    print(f"\n{'='*60}")
    print("BEAM MATRIX GENERATION STARTED")
    print(f"nside: {nside}")
    print(f"Input file: {path_to_file}")
    print(f"Fields to process: {fields}")
    if observed_pixels_file:
        print(f"Observed pixels file: {observed_pixels_file}")
    else:
        print("Using full sky coverage")
    print(f"{'='*60}\n")
    
    main(nside, path_to_file, fields, observed_pixels_file)