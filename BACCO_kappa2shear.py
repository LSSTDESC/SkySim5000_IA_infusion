import numpy as np
import healpy as hp
import h5py 


path_in = '/home/dc-harn1/rds/rds-dirac-dp323-PcgcjiGzdYk/BACCO_fullsky/lightcones/lc_LH0/0.00/obs_25234/'

nshells = 35
a_maps = np.linspace(1/(1+3.0), 1.0,nshells+1, dtype = np.float32)
rds = np.flip(1.0/a_maps - 1.0)
rds_upper = rds[1:nshells+1]
nside=4096

# Compute C_ell:
#for zs in rds_upper[0:nshells]:
for zs in rds_upper[[19, 27, 34]]:
#for zs in rds_upper[0:nshells]:

    print( "\nLoading convergence map and computing shear maps for zs={:5.4f}\n".format(zs) )
    fname = path_in+'/kappa/'+str('{:5.4f}').format(zs)+'kappa.npy'
    kpmap_ring = np.load(fname)
    print("loaded kappa")

    # Compute shear:
    print("starting kappa map2alm")
    alm_E = hp.sphtfunc.map2alm(kpmap_ring, lmax=2*nside, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
    #alm_E = hp.sphtfunc.map2alm(kpmap_ring, lmax=4*nside, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
    alm_B = alm_E*0.0;
    #

    #-------------------------------
    print("Starting shear alm2map")
    maps_QU = hp.alm2map_spin((alm_E, alm_B), nside, spin=2,lmax=2*nside,mmax=None)
    #maps_QU = hp.alm2map_spin((alm_E, alm_B), nside, spin=2,lmax=4*nside,mmax=None)

    #
    gamma1 = maps_QU[0]
    gamma2 = maps_QU[1]
    print("Got shear")

    fname1 = path_in+'/shear/'+str('{:5.4f}').format(zs)+'gamma1'
    fname2 = path_in+'/shear/'+str('{:5.4f}').format(zs)+'gamma2'

    np.save(fname1, gamma1)
    np.save(fname2, gamma2)

