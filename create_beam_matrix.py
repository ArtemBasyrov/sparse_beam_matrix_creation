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

    filename = "beam_sparse_{0}_FWHM{1}.npz".format(nside, FWHM)
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