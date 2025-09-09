import numpy as np
import healpy as hp
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
import argparse


def theta_from_querry_disk(nside, FWHM):
    npix = hp.nside2npix(nside)
    vec = hp.pix2vec(nside, np.arange(npix))
    vec = np.array(vec).T  # transpose to get shape (3, npix)
    lon, lat = hp.pix2ang(nside, np.arange(npix), lonlat=True)

    data = []
    row = []
    col = []

    FWHM_rad = FWHM * np.pi / (180. * 60.)  # convert FWHM to radians

    for i in range(npix):
        ipix_neighbours = hp.query_disc(nside, vec[i], 3*FWHM_rad, inclusive=True)
        ang_dist = hp.rotator.angdist((lon[i], lat[i]), (lon[ipix_neighbours], lat[ipix_neighbours]), lonlat=True)
        
        data += list(ang_dist)
        row += [i] * len(ipix_neighbours)
        col += list(ipix_neighbours)

    sparse_theta_ij_query = sp.csr_array((data, (row, col)))
    return sparse_theta_ij_query


def beam_sparse(theta_sparse, FWHM):
    """Gaussian beam function."""
    FWHM = FWHM / 180. * np.pi /60.  # convert FWHM from arcmin to radians
    sigma = FWHM / (2. * np.sqrt(2. * np.log(2.)))  # convert FWHM to sigma
    B_data = np.exp(-0.5 * (theta_sparse.data / sigma) ** 2)

    B_ij = theta_sparse.copy()
    B_ij.data = B_data 

    row_sum = B_ij.sum(axis=1).flatten()
    norm_matrix = sp.diags(1.0 / row_sum, format='csr')
    B_ij = norm_matrix @ B_ij  # normalize each row

    return B_ij


def find_best_fit_a(nside, tol=1e-6):
    """
    Find the best fit parameter 'a' for the Gaussian beam that minimizes the difference
    between bl² and the pixel window function.
    
    Parameters:
    - nside: int, HEALPix resolution parameter
    - tol: float, tolerance for optimization (default=1e-6)
    
    Returns:
    - best_a: float, optimal value of parameter 'a'
    """
    lmax = 2*nside # from testing this is the best option

    # Harmonic pixel window function
    pixwin = hp.sphtfunc.pixwin(nside, pol=False, lmax=lmax)

    # Define the objective function to minimize
    def objective(a):
        bl = hp.gauss_beam(hp.nside2resol(nside)/a, lmax=lmax)
        diff = np.sum((bl - pixwin)**2)
        return diff
    
    # Find the optimal 'a' using bounded optimization
    result = minimize_scalar(objective, bounds=(0.1, 10), method='bounded', options={'xatol': tol})
    
    return result.x


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


def main(nside, FWHM):
    # select the larger of the two: beam FWHM or pixel resolution
    res_arcmin = hp.nside2resol(nside, arcmin=True)
    if FWHM > res_arcmin:
        sparse_theta_ij_query = theta_from_querry_disk(nside, FWHM)
    else:
        sparse_theta_ij_query = theta_from_querry_disk(nside, res_arcmin)
    print("Created sparse theta_ij matrix.")

    # create B' = PB from the beam and pixel window matrices
    B_ij_sm = beam_sparse(sparse_theta_ij_query, FWHM=FWHM) 
    a = find_best_fit_a(nside)
    B_ij_pix = beam_sparse(sparse_theta_ij_query, FWHM=res_arcmin/a) # approxiamte the pixel window matrix with a gaussian beam
    B_ij = B_ij_pix.dot(B_ij_sm)

    # eliminate very small values to save space
    B_ij.data[B_ij.data < 1e-15] = 0 # np.float64 precision
    B_ij.eliminate_zeros()

    # equalize the number of non-zero elements in each row
    N_cutoff = np.diff(B_ij.indptr)
    N_neighb = np.min(N_cutoff)
    B_ij = keep_top_n_neighb(B_ij, N_neighb)

    filename = "beam_sparse_{0}_FWHM{1}_cutoff.npz".format(nside, FWHM)
    sp.save_npz(filename, B_ij)
    print(f"Saved beam matrix to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create beam matrix for given nside and FWHM.")
    parser.add_argument("--nside", type=int, help="HEALPix nside parameter.")
    parser.add_argument("--fwhm", type=float, help="Beam FWHM in arcminutes.")

    args = parser.parse_args()
    nside = args.nside
    FWHM = args.fwhm

    main(nside, FWHM)