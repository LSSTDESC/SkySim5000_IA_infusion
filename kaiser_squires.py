# This module contains the code for a simple flat Kaiser-Squires inversion
import numpy as np
import healpy as hp

def flat_KS_map(gmap):
    """Compute kappa maps from binned shear maps.
    returns kappa_e and kappa_b
    """
    nx = gmap.shape[1]
    ny = gmap.shape[2]

    # get frequencies
    k1, k2 = np.meshgrid(np.fft.fftfreq(nx),
                         np.fft.fftfreq(ny))

    g1 = np.fft.fft2(gmap[0])
    g2 = np.fft.fft2(gmap[1])
    denom = k1*k1 + k2*k2
    denom[0, 0] = 1  # avoid division by 0
    kap = ((k1*k1 - k2*k2) - 2j*(k1*k2)) * (g1 + 1j*g2) / denom
    kap = np.fft.ifft2(kap)

    return np.real(kap), np.imag(kap)

def healpix_KS_map(gmap, lmax=None, sigma=None):
    """
    Computes kappa maps from a given healpix shear map (in ring format)
    Adapted from `g2k_sphere` DES code:
    https://github.com/chihway/massmapping/blob/master/utils/massmapping_utils.py

    Parameters
    ----------
    gmap: ndarray
        Healpix convergence map (ring format)
    lmax: int
        Maximum multipole order
    sigma: float
        Gaussian smoothing applied to the alms [arcmin]
    """
    nside = hp.npix2nside(gmap.shape[1])

    if sigma is not None:
        # convert to radians
        sigma = sigma / 60./180*np.pi

    if lmax is None:
        lmax = 2*nside

    KQU_maps = [np.zeros_like(gmap[0]), gmap[0], gmap[1]]
    alms = hp.map2alm(KQU_maps, lmax=lmax, pol=True)

    ell, emm = hp.Alm.getlm(lmax=lmax)
    almsE = alms[1]*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
    almsB = alms[2]*((ell*(ell+1.))/((ell+2.)*(ell-1)))**0.5
    almsE[ell==0] = 0.0
    almsB[ell==0] = 0.0
    almsE[ell==1] = 0.0
    almsB[ell==1] = 0.0

    E_map = hp.alm2map(almsE, nside=nside, lmax=lmax, pol=False, sigma=sigma, verbose=False)
    B_map = hp.alm2map(almsB, nside=nside, lmax=lmax, pol=False, sigma=sigma, verbose=False)
    return E_map, B_map
