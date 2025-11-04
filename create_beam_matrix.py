import numpy as np
import healpy as hp
import scipy.sparse as sp
import argparse


def theta_from_querry_disk(nside, FWHM, observed_pixels=None):
    npix = hp.nside2npix(nside)
    
    if observed_pixels is None:
        observed_pixels = np.arange(npix)
    
    vec = hp.pix2vec(nside, observed_pixels)
    vec = np.array(vec).T
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)
    set2 = set(observed_pixels)

    data = []
    row = []
    col = []

    FWHM_rad = FWHM * np.pi / (180. * 60.)

    for i, pix_idx in enumerate(observed_pixels):
        ipix_neighbours = hp.query_disc(nside, vec[i], 3*FWHM_rad, inclusive=False)

        # Keep only neighbors that are in observed_pixels
        set1 = set(ipix_neighbours)
        ipix_neighbours = np.array(list(set1 & set2))

        ang_dist = hp.rotator.angdist((lon[pix_idx], lat[pix_idx]), 
                                    (lon[ipix_neighbours], lat[ipix_neighbours]), 
                                    lonlat=True)
        
        data += list(ang_dist)
        row += [i] * len(ipix_neighbours)
        col += list(ipix_neighbours)

    # Create sparse matrix with shape (n_observed, npix)
    sparse_theta_ij_query = sp.csr_array((data, (row, col)), 
                                        shape=(len(observed_pixels), npix))
    return sparse_theta_ij_query, observed_pixels


def beam_sparse(theta_sparse, FWHM):
    """Gaussian beam function."""
    FWHM = FWHM / 180. * np.pi / 60.
    sigma = FWHM / (2. * np.sqrt(2. * np.log(2.)))
    B_data = np.exp(-0.5 * (theta_sparse.data / sigma) ** 2)

    B_ij = theta_sparse.copy()
    B_ij.data = B_data 

    row_sum = B_ij.sum(axis=1).flatten()
    norm_matrix = sp.diags(1.0 / row_sum, format='csr')
    B_ij = norm_matrix @ B_ij

    return B_ij


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
    

def main(nside, FWHM, observed_pixels_file=None):
    # Load observed pixels if provided
    if observed_pixels_file:
        observed_pixels = np.load(observed_pixels_file)
        print(f"Loaded {len(observed_pixels)} observed pixels from {observed_pixels_file}")
    else:
        observed_pixels = None
        print("Using full sky coverage")

    # Find theta_ij angular distances
    res_arcmin = hp.nside2resol(nside, arcmin=True)
    if FWHM > res_arcmin:
        sparse_theta_ij_query, observed_pixels = theta_from_querry_disk(nside, FWHM, observed_pixels)
    else:
        sparse_theta_ij_query, observed_pixels = theta_from_querry_disk(nside, res_arcmin, observed_pixels)
    print(f"Created sparse theta_ij matrix with shape {sparse_theta_ij_query.shape}")

    B_ij = beam_sparse(sparse_theta_ij_query, FWHM=FWHM)
    
    # Eliminate very small values to save space
    B_ij.data[B_ij.data < 1e-15] = 0
    B_ij.eliminate_zeros()

    # Pad to maximum number of neighbors per pixel
    data_array, indices_array, max_nnz = pad_to_max_neighbors(B_ij)
    print(f"Max neighbors per pixel: {max_nnz}")
    
    filename = "beam_sparse_{0}_FWHM{1}.npz".format(nside, FWHM)
    save_beam_matrix(filename, data_array, indices_array, observed_pixels, nside, max_nnz)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create beam matrix for given nside and FWHM.")
    parser.add_argument("--nside", type=int, required=True, help="HEALPix nside parameter.")
    parser.add_argument("--fwhm", type=float, required=True, help="Beam FWHM in arcminutes.")
    parser.add_argument("--observed_pixels", type=str, help="Optional: .npy file with observed pixel indices")

    args = parser.parse_args()
    main(args.nside, args.fwhm, args.observed_pixels)