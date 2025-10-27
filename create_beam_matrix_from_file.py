import numpy as np
import healpy as hp
import scipy.sparse as sp
from scipy.spatial.transform import Rotation as R
import argparse
from pixell import enmap
from numba import jit
import numba as nb
import pandas as pd

def read_beam_file(path_to_file, field=0):
    beam_map = enmap.read_map(path_to_file)
    pixel_map = np.array(beam_map[field]).copy()

    ra, dec = beam_map.posmap()
    ra = ra * 180/np.pi * 60    # convert to arcmin
    dec = dec * 180/np.pi * 60  # convert to arcmin

    # center the map at (0,0)
    ix, iy = (100,100)
    dec_c = dec[ix,iy]
    ra_c = ra[ix,iy]

    dec = dec - dec_c
    ra = ra - ra_c

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
    if total > 0:
        beam_stacked /= total
    
    return beam_stacked

def recenter_coordinates_vector_batch(ra, dec, phi_pix_batch, theta_pix_batch, center_idx=(100, 100), sel=None):
    """Vectorized version for batch processing"""
    def spherical_to_vector(phi, theta):
        x = np.sin(theta) * np.cos(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(theta)
        return np.stack([x, y, z], axis=-1)
    
    def vector_to_spherical(vec):
        x, y, z = vec[..., 0], vec[..., 1], vec[..., 2]
        theta = np.arccos(np.clip(z, -1, 1))  # Avoid numerical errors
        phi = np.arctan2(y, x)
        phi = np.where(phi < 0, phi + 2*np.pi, phi)
        return phi, theta
    
    # Convert input coordinates to vectors
    phi_orig = np.radians(ra / 60.0)
    theta_orig = np.pi/2 - np.radians(dec / 60.0)
    vec_orig = spherical_to_vector(phi_orig, theta_orig)
    
    # Apply selection mask if provided
    if sel is not None:
        vec_orig_sel = vec_orig[sel].reshape(-1, 3)
    else:
        vec_orig_sel = vec_orig.reshape(-1, 3)
    
    # Batch process rotations
    phi_new_batch = []
    theta_new_batch = []
    
    vec_center = spherical_to_vector(phi_orig[center_idx], theta_orig[center_idx])
    
    for phi_pix, theta_pix in zip(phi_pix_batch, theta_pix_batch):
        vec_target = spherical_to_vector(phi_pix, theta_pix)
        
        if np.allclose(vec_center, vec_target):
            vec_rotated = vec_orig_sel
        else:
            axis = np.cross(vec_center, vec_target)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(vec_center, vec_target), -1, 1))
            rot = R.from_rotvec(axis * angle)
            vec_rotated = rot.apply(vec_orig_sel)
        
        phi_new, theta_new = vector_to_spherical(vec_rotated)
        phi_new_batch.append(phi_new)
        theta_new_batch.append(theta_new)
    
    return phi_new_batch, theta_new_batch

def beam_from_file(nside, FWHM, path_to_file, field=0, batch_size=1000):
    npix = hp.nside2npix(nside)
    theta_pix, phi_pix = hp.pix2ang(nside, np.arange(npix))

    data = []
    row = []
    col = []

    FWHM_rad = FWHM * np.pi / (180. * 60.)
    pixel_map, ra, dec = read_beam_file(path_to_file, field=field)
    sel = (ra**2 + dec**2) < (3*FWHM)**2
    pixel_map_sel = pixel_map[sel]
    
    for i_start in range(0, npix, batch_size):
        i_end = min(i_start + batch_size, npix)
        batch_indices = np.arange(i_start, i_end)
        
        # Vectorize coordinate transformation for batch
        phi_batch, theta_batch = recenter_coordinates_vector_batch(
            ra, dec, phi_pix[batch_indices], theta_pix[batch_indices], 
            center_idx=(100, 100), sel=sel
        )
        
        for idx, i in enumerate(batch_indices):
            theta = theta_batch[idx]
            phi = phi_batch[idx]
            
            try:
                pix_patch = hp.ang2pix(nside, theta, phi)
            except ValueError as e:
                print(f"Pixel index {i} caused an error in ang2pix: {e}")
                print(f"theta: {theta}, phi: {phi}")
                continue
            
            ipix_neighbours = np.unique(pix_patch)
            pix_sel = hp.query_disc(nside, hp.pix2vec(nside, i), 3*FWHM_rad, inclusive=False)
            ipix_neighbours = fast_intersect1d_numba(ipix_neighbours, pix_sel)

            if len(ipix_neighbours) == 0:
                continue

            beam_stacked = beam_stacking_optimized(pixel_map_sel, pix_patch, ipix_neighbours)

            # Only add if we have valid beam values
            if np.any(beam_stacked > 0):
                data.extend(beam_stacked)
                row.extend([i] * len(ipix_neighbours))
                col.extend(ipix_neighbours)

            if i % 100000 == 0:
                print(f"Processed pixel {i}/{npix}")

    sparse_beam_ij_query = sp.csr_array((data, (row, col)), shape=(npix, npix))
    return sparse_beam_ij_query

def keep_top_n_neighb(B, N_neighb):
    """
    Keep only the top N_neighb largest elements in each row of the sparse matrix B.
    Parameters:
    - B: scipy.sparse matrix, input sparse matrix
    - N_neighb: int, number of largest elements to keep per row
    Returns:
    - result: scipy.sparse matrix, sparse matrix with only top N_neighb elements per row
    """
    if not sp.isspmatrix_csr(B):
        B = B.tocsr()
    
    result = B.copy()
    
    # For each row, find the threshold value
    thresholds = np.zeros(B.shape[0])
    
    for i in range(B.shape[0]):
        start, end = B.indptr[i], B.indptr[i+1]
        row_size = end - start
        
        if row_size <= N_neighb:
            thresholds[i] = -np.inf  # Keep all elements
        else:
            row_data = B.data[start:end]
            partitioned = np.partition(row_data, -N_neighb) # Find the N_neighb-th largest value
            thresholds[i] = partitioned[-N_neighb]
    
    # Apply threshold to all rows
    for i in range(B.shape[0]):
        start, end = B.indptr[i], B.indptr[i+1]
        row_data = result.data[start:end]
        keep_mask = row_data >= thresholds[i]

        if len(keep_mask[keep_mask]) > N_neighb:
            # In case of ties, ensure only N_neighb elements are kept
            sorted_indices = np.argsort(-row_data)
            keep_mask = np.zeros_like(row_data, dtype=bool)
            keep_mask[sorted_indices[:N_neighb]] = True

        result.data[start:end] = np.where(keep_mask, row_data, 0)
    
    result.eliminate_zeros()
    return result

def main(nside, FWHM, path_to_file):
    # select the larger of the two: beam FWHM or pixel resolution
    res_arcmin = hp.nside2resol(nside, arcmin=True)
    field = 1 # U Stokes parameter
    if FWHM > res_arcmin:
        B_ij = beam_from_file(nside, FWHM, path_to_file, field=field)
    else:
        B_ij = beam_from_file(nside, res_arcmin, path_to_file, field=field)
    print("Created sparse beam_ij matrix.")

    # eliminate very small values to save space
    B_ij.data[B_ij.data < 1e-15] = 0 # np.float64 precision
    B_ij.eliminate_zeros()

    # equalize the number of non-zero elements in each row
    N_cutoff = np.diff(B_ij.indptr)
    N_neighb = np.min(N_cutoff)
    B_ij = keep_top_n_neighb(B_ij, N_neighb)
    print(f"Created beam matrix with {N_neighb} non-zero elements per row.")

    filename = "beam_sparse_{0}_FWHM{1}_cutoff_satp1_ws0_Q.npz".format(nside, FWHM)
    sp.save_npz(filename, B_ij)
    print(f"Saved beam matrix to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create beam matrix for given nside and FWHM.")
    parser.add_argument("--nside", type=int, help="HEALPix nside parameter.")
    parser.add_argument("--fwhm", type=float, help="Beam FWHM in arcminutes.")
    parser.add_argument("--file", type=str, help="Path to pixel map FITS file.")

    args = parser.parse_args()
    nside = args.nside
    FWHM = args.fwhm
    path_to_file = args.file

    main(nside, FWHM, path_to_file)