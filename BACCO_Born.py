#!/usr/bin/env python
# coding: utf-8

# In[39]:


#/usr/bin/env python
import numpy as np
import pylab as pl
import subprocess as sp
#import source_redshift_dist as srd
import healpy as hp
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import input
import os
from scipy.integrate import quad
from scipy.interpolate import interp1d
import healsparse
import random
import h5py


ncosmo = FlatLambdaCDM(H0=66.99998481005214,Om0=0.328999999, Ob0=0.049000)


rho_crit = ncosmo.critical_density0.to(u.solMass/u.Mpc/u.Mpc/u.Mpc).value/ncosmo.h/ncosmo.h #M_sun Mpc^-3 *h*h
vc = 2.998e5 #km/s
G = 4.3011790220362e-09 # Mpc/h (Msun/h)^-1 (km/s)^2
mp      = 1.0*10**9*ncosmo.h     # in Msun/h

#From Cyrille: HACC: 2602746430.321433


########################### Mass Maps ################################
path_in = '/global/cfs/cdirs/m5099/FS2/'
path_out = '/pscratch/sd/j/jharno/FS/'





# In[33]:


def total_area(x_in,y_in):
    x_down = x_in[:-1]
    x_up = x_in[1:]
    y_down = y_in[:-1]
    y_up = y_in[1:]

    area_array = (y_down+y_up)*(x_up-x_down)/2.0

    return np.sum(area_array)


def cal_pdz(fname, z_in,col):

    #zs,dist1 = np.loadtxt("./data/source_distribution.txt",comments='#',usecols=(0,1),unpack=True)
    zs,dist1 = np.loadtxt(fname,comments='#',usecols=(0,col),unpack=True)
    dist1_normal = dist1/total_area(zs,dist1)
    # dist1_normal = dist1

    #np.savetxt("./data/srd_normed.dat",np.transpose(np.array([zs,dist1_normal])),fmt="%.6e")

    # f1 = interp1d(zs[::-1], dist1_normal[::-1], kind='cubic',bounds_error=False, fill_value=0.0)
    f1 = interp1d(zs[::-1], dist1_normal[::-1], kind='linear',bounds_error=False, fill_value=0.0)
    #f1 = interp1d(zs, dist1_normal, kind='linear',fill_value="extrapolate")

    return f1(z_in)

def wfunc_all(fname, zl,col):

    if(zl==0):
        zl=0.0001
        print("set minimal zl to 0.0001")
    Dc_l = ncosmo.comoving_distance(zl).value*ncosmo.h
    cfactor = 4.0*np.pi*G/vc**2*(1.0+zl)/Dc_l


    zs_list = np.linspace(0.001,4.0,1000)
    #zs_list = np.logspace(-3,0.6,100)
    dzs = zs_list[1]-zs_list[0]
    total_wfunc = 0.0

    for zs in zs_list:
        if (zs<zl):
            continue

        Dc_s = ncosmo.comoving_distance(zs).value*ncosmo.h
        Dc_ls = Dc_s - Dc_l

        tmp = dzs*cal_pdz(fname, zs,col)*Dc_ls/Dc_s
        total_wfunc = total_wfunc + tmp

    #res = cfactor/afactor*total_wfunc
    res = cfactor*total_wfunc

    return res

def kappa_sl_ss(filename,zl,zs):

    if(zl==0):
        zl=0.005
        print("set minimal zl to 0.005")

    Dc_l = ncosmo.comoving_distance(zl).value*ncosmo.h
    Dc_s = ncosmo.comoving_distance(zs).value*ncosmo.h
    Dc_ls = Dc_s - Dc_l

    sdens=np.load(filename)
    # turn masked pixels to zero.
    #sdens[sdens<-1.0e+30]=0

    #Map already in ring format:




    sdens_arc = sdens#/np.deg2rad(1.0)**2.0#*3.0462e-4 # 1 degree^2 = 3.0462e-4 steradian
    sdens_arc*=mp
    sdens_arc_normal = (sdens_arc-sdens_arc.mean()) 

    #Normalisation:
    Norm = 1.6125e+07


    #-----------------------
    # Results 2
    cfactor = 4.0*np.pi*G/vc**2
    res = cfactor*(1.0+zl)/Dc_l*Dc_ls/Dc_s*sdens_arc_normal*Norm
    #print('Kernel = ', zl, zs, cfactor*(1.0+zl)/Dc_l*Dc_ls/Dc_s)
    return res

def kappa_sl_ms(filename,fname,zl,col):

    nside_c = 32    
    a = healsparse.HealSparseMap.read(os.path.join(path, filename))
    sdens = a.generate_healpix_map()

    # turn masked pixels to zero.
    sdens[sdens<-1.0e+30]=0
    #Map already in ring format:
    #hp.mollview(sdens,
    #            unit='kappa', cmap=pl.cm.coolwarm,
    #            cbar=True, title='', norm=None
    #            , nest=False
    #            )
    #pl.show()



    sdens_arc = sdens#/np.deg2rad(1.0)**2.0#*3.0462e-4 # 1 degree^2 = 3.0462e-4 steradian
    sdens_arc*=mp
    sdens_arc_normal = (sdens_arc-sdens_arc.mean()*8.0) # factor of 8.0 corrects for mean computed over all 8 octants

    #Normalisation:
    Norm = 1.6125e+07

    res = wfunc_all(fname, zl,col)*sdens_arc_normal*Norm

    print(filename, zl, mp, wfunc_all(fname, zl,col))
    #hp.mollview(res,
    #            unit='kappa_z', cmap=pl.cm.coolwarm,
    #            cbar=True, title='', norm=None
    #            , nest=False
    #            )
    #pl.show()

    return res



# In[34]:


# Set redshift arrays:
#zfile=np.loadtxt("z2ts_HACCY1_"+str(nshells)+".txt")


nshells=200 # update to 51 soon
zfile = np.loadtxt(path_in+"steps.fs2.ssv",skiprows=1)
# step dist_min dist_max dist_median z_median a_median
# distances are in Mpc/h

snaplist = zfile[0:nshells,0].astype(int)
zsnap=zfile[0:nshells,4]
#NOTE: in FS, high-redshift is low snap, and vice versa. Just likein OuterRim, but opposite to HACC.

#print(snaplist[0:nshells])
#print(zsnap)

chi_lower_Mpc = zfile[0:nshells,1]/ ncosmo.h
chi_upper_Mpc = zfile[0:nshells,2]/ ncosmo.h


rds_lower[0]=0.0
rds_lower[1:200] = z_at_value(ncosmo.comoving_distance,chi_lower_Mpc[1:200]*u.Mpc).value

rds_upper = z_at_value(ncosmo.comoving_distance,chi_upper_Mpc*u.Mpc).value
#print(chi_upper*u.Mpc, z_at_value(ncosmo.comoving_distance,chi_upper*u.Mpc).value)# 
#print(rds_upper)

# TO FIX:
# MUST INCLUDE THE ***dz_dchi***


#-----
# Get lens redshifts, either from geometry (Wrong, I miss the dNdz.):

zlens_exact=np.zeros_like(zsnap)
for plane in range(0,nshells):
    #zlens_exact[plane]=zsnap[nshells-1-plane]
    zlens_exact[plane]=zsnap[plane]
zlens_exact = zlens_exact[:nshells] 

print( " z_       z^                           zl_file            "  )
for plane in range(0,nshells):
    print(rds_lower[plane], rds_upper[plane],zlens_exact[plane] )


#print(zlens_exact)


# In[43]:


# Mass2Kappa_zslice
# For n(z) see further below;

nside = 1024
npix = 12 * nside**2
noct = npix / 8 - nside

ApplyMask=False

#----------------------------------
# Start loop over source redshifts

#for zs in zsnap[1:]:
for zs in rds_upper[0:nshells]:
#for zs in rds_upper[0:1]:


    # Initialise kappa maps:
    kpmap = np.zeros(npix)

    #planeIndexes = dlup[:,0]<=zs
    #planeIndexes = np.flip(zlens_exact)<=zs
    #print(zs,planeIndexes)

    for lens in range(0,nshells):
        zl=zlens_exact[lens]

        if zl >= zs:          
            continue

        #lens_str = str(lens).zfill(2)
        #filename= 'nshells_'+str(nshells)+'_imap_{}/baryonified_shells_v11.h5'.format(lens_str)

        filename = 'counts_'+str(nshells-lens)+'_1024.npy'
        print(filename, lens, zl, zs)

        kappa_z = kappa_sl_ss(path_in+filename, zl, zs)
        kpmap = kpmap + kappa_z
        print("Got lens " +str(zl))
        #print(kpmap)

    # Convert to ring-ordered for sph.harmonic transforms:
    #kpmap_ring = hp.reorder(kpmap, n2r = True)
    #print("reordered map")

    #----------------------------
    #Apply mask?
    if(ApplyMask):
        if 'hpmap_weight' in locals():
            print("Mask already defined!")
        else:
            print("Reading in weight map for Mira Titan (Comment if already on disk...)")
            hpmap_weight = hp.fitsfunc.read_map('../..//FullSky_weight.fits', field=0,nest=False, partial=False, hdu=1, h=False, verbose=True, memmap=True)
            #hpmap_weight = hp.fitsfunc.read_map('../../FullSky_weight_4096.fits', field=0,nest=False, partial=False, hdu=1, h=False, verbose=True, memmap=True)

        # show mask:
        #hp.mollview(hpmap_weight,
        #            unit=None, cmap=pl.cm.coolwarm,
        #            cbar=True, title='weight', norm=None
        #            , nest=False
        #            )
        #pl.show()

        kpmap *= hpmap_weight
        #print("applied mask")

    # show kappa:
    #np.max(kpmap_ring)
    hp.mollview(kpmap,
                unit='kappa', cmap=pl.cm.coolwarm,
                cbar=True, title='', norm=None
                , nest=False
                )
    pl.show()

    lonra = [0, 20]
    latra = [0, 10]

    hp.cartview(kpmap,
                cbar=True, lonra=lonra, latra=latra,
                nest=False,
                title=" ",
                # min=-10, max=0,
                cmap=pl.cm.coolwarm,
                norm=None, unit='kappa')
    pl.show()

    # save to file:
    fname = path_out+'/kappa/'+str('{:5.4f}').format(zs)+'kappa'

    np.save(fname, kpmap)
    print("Saved kappa")


# In[45]:


# Compute C_ell:
for zs in rds_upper[0:200]:

    print( "\nLoading convergence map and computing C_ell's: zs={:5.4f}\n".format(zs) )
    fname = path_out+'/kappa/'+str('{:5.4f}').format(zs)+"kappa.npy"
    kpmap = np.load(fname)
    print("loaded kappa")

    lonra = [0, 5]
    latra = [0, 2]
    hp.cartview(kpmap,
            cbar=True, lonra=lonra, latra=latra,
            nest=False,
            title=" ",
            # min=-10, max=0,
            cmap=pl.cm.coolwarm,
            norm=None, unit='kappa')
    pl.show()
    #print(np.shape(kpmap))
    #binwidth = 0.05
    #(n1, bins1, patches1) = pl.hist(kpmap, facecolor = 'blue')

    pl.show()

    # Compute C_ell:
    Cl = hp.sphtfunc.anafast(kpmap, map2=None, nspec=None, lmax=4*nside, mmax=None, iter=1, alm=False, pol=False, use_weights=False, datapath=None, gal_cut=0, use_pixel_weights=False)
    print("Got C_ell")
    fname = path_out+'/kappa/'+str("{:5.4f}".format(zs))+"Cl.dat"
    np.savetxt(fname,Cl)
    print("Saved C_ell FS")



# In[47]:


# kappa2shear:

nside = 1024
npix = 12 * nside**2
noct = npix / 8 - nside
seed=1
#IC=150


for zs in rds_upper[0:nshells]:

    print( "\nLoading convergence map and computing shear maps for zs={:5.4f}\n".format(zs) )
    fname = path_out+'/kappa/'+str('{:5.4f}').format(zs)+'kappa.npy'
    kpmap_ring = np.load(fname)
    print("loaded kappa")

    # Compute shear:
    print("starting kappa map2alm")
    alm_E = hp.sphtfunc.map2alm(kpmap_ring, lmax=4*nside, mmax=None, iter=3, pol=False, use_weights=False, datapath=None)
    alm_B = alm_E*0.0;
    #

    #-------------------------------
    print("Starting shear alm2map")
    maps_QU = hp.alm2map_spin((alm_E, alm_B), nside, spin=2,lmax=4*nside,mmax=None)

    #
    gamma1 = maps_QU[0]
    gamma2 = maps_QU[1]
    print("Got shear")

    fname1 = path_out+'/shear/'+str('{:5.4f}').format(zs)+'gamma1'
    fname2 = path_out+'/shear/'+str('{:5.4f}').format(zs)+'gamma2'

    np.save(fname1, gamma1)
    np.save(fname2, gamma2)
    print("Saved gamma")



# In[ ]:


# Mass2Kappa_n(z)
nside = 4096
npix = 12 * nside**2
noct = npix / 8 - nside

#nshells=19
seed=0
IC=150
path = '/global/cscratch1/sd/xuod/HOS_sims/L845/HACC'+str(IC)+'/'


#input_dir = input.baseDirectory
#mapsNames = sorted(directory_contents)[58:1:-1] # use this one when there is a README file in the directory.
#mapsNames = sorted(directory_contents)[57:0:-1]

#mapsNames = np.array(mapsNames)
#print("Found these maps in the directory:", mapsNames)
#print("Found a total of ",np.shape(mapsNames), "Maps")
#print("*****")


# Start loop over source redshifts

for tomo in range(1,6):

    print("Working on tomo", tomo)
    # Initialise kappa maps:
    kpmap = np.zeros(npix)
    zid=0

    for lens in range(0,nshells):
        zl=zlens_exact[zid]

        filename= 'shells_z'+str(nshells)+'_subsampleauto_groupiso/shell_hacc_z'+str(lens)+'_nside4096_seed'+str(seed)+'.fits'
        print(filename, lens, zl, zid)

        kappa_z = kappa_sl_ms(path+filename,"./Euclid_dndz_fu08_bin1-5.dat",zl,tomo)
        kpmap = kpmap + kappa_z
        print("Got lens " +str(zl))
        print(kappa_z)
        zid+=1
        #hp.cartview(kpmap,
        #        cbar=True, lonra=lonra, latra=latra,
        #        nest=False,
        #        title=" ",
        #        # min=-10, max=0,
        #        cmap=pl.cm.coolwarm,
        #        norm=None, unit='kappa')
        #pl.show()

    #Map already in ring format, no need to re-order

    #Apply mask
    if 'hpmap_weight' in locals():
        print("Mask already defined!")
    else:
        print("Reading in weight map for Mira Titan (Comment if already on disk...)")
        #hpmap_weight = hp.fitsfunc.read_map('../../FullSky_weight.fits', field=0,nest=False, partial=False, hdu=1, h=False, verbose=True, memmap=True)
        hpmap_weight = hp.fitsfunc.read_map('../../FullSky_weight_4096.fits', field=0,nest=False, partial=False, hdu=1, h=False, verbose=True, memmap=True)

    # show mask:
    #hp.mollview(hpmap_weight,
    #            unit=None, cmap=pl.cm.coolwarm,
    #            cbar=True, title='weight', norm=None
    #            , nest=False
    #            )
    #pl.show()

    kpmap *= hpmap_weight
    print("applied mask")

    # show kappa:
    #np.max(kpmap_ring)
    #hp.mollview(kpmap,
    #            unit='kappa', cmap=pl.cm.coolwarm,
    #            cbar=True, title='', norm=None
    #            , nest=False
    #            )
    #pl.show()

    #lonra = [0, 20]
    #latra = [0, 10]

    #hp.cartview(kpmap,
    #            cbar=True, lonra=lonra, latra=latra,
    #            nest=False,
    #            title=" ",
    #            # min=-10, max=0,
    #            cmap=pl.cm.coolwarm,
    #            norm=None, unit='kappa')
    #pl.show()

    # save to file, in .npy format:
    #fname = input.kappaDir+"kappa_tomo"+np.str(tomo)
    fname = "../../../HACC-Y1/kappa/V0/IC"+str(IC)+"/seed"+str(seed)+"/kappa_tomo"+np.str(tomo)+"_"+np.str(nshells)

    np.save(fname, kpmap)
    print("Saved kappa")


# In[ ]:


# Compute C_ell:
for tomo in range(1,6):

    print( "\nLoading convergence map and computing C_ell's: tomo="+np.str(tomo))
    fname = "../../../HACC-Y1/kappa/V0/IC"+str(IC)+"/seed"+str(seed)+"/kappa_tomo"+np.str(tomo)+"_"+str(nshells)+".npy"
    kpmap_ring = np.load(fname)
    print("loaded kappa")

    # Compute C_ell:
    Cl = hp.sphtfunc.anafast(kpmap_ring, map2=None, nspec=None, lmax=5000, mmax=None, iter=1, alm=False, pol=False, use_weights=False, datapath=None, gal_cut=0, use_pixel_weights=False)
    Cl *= 8.0
    print("Got C_ell")
    fname = "../../../HACC-Y1/kappa/V0/IC"+str(IC)+"/seed"+str(seed)+"/Cl_tomo"+np.str(tomo)+"_"+str(nshells)+".dat"
    np.savetxt(fname,Cl)
    print("Saved C_ell")



# In[ ]:


#hpmap_weight = hp.fitsfunc.read_map('../../FullSky_weight.fits', field=0,nest=False, partial=False, hdu=1, h=False, verbose=True, memmap=True)

#show mask:
#hp.mollview(hpmap_weight,
#            unit=None, cmap=pl.cm.coolwarm,
#            cbar=True, title='weight', norm=None
#            , nest=False
#            )
#pl.show()
#print(hpmap_weight)

#hp.mollview(kpmap,
#            unit='kappa', cmap=pl.cm.coolwarm,
#            cbar=True, title='', norm=None
#            , nest=False
#            )
#pl.show()
#print(kpmap)


hp.cartview(kpmap_ring,
            cbar=True, lonra=lonra, latra=latra,
            nest=False,
            title=" ",
            # min=-10, max=0,
            cmap=pl.cm.coolwarm,
            norm=None, unit='kappa')
pl.show()
print(kpmap_ring)


# In[ ]:


#Construct mask with correct NSIDE & quadrant:
#a = healsparse.HealSparseMap.read(os.path.join(path, filename))
#mask = a.generate_healpix_map()

# turn masked pixels to zero.
#mask[mask>-1.0e+30]=1
#mask[mask<-1.0e+30]=0

#hp.mollview(mask,
#            unit=None, cmap=pl.cm.coolwarm,
#            cbar=True, title='weight', norm=None
#            , nest=False
#            )
#pl.show()
#hp.fitsfunc.write_map('../../FullSky_weight_4096.fits',mask,nest=False,overwrite=True)


# In[ ]:


#del hpmap_weight


# In[ ]:


# Mass2Kappa_n(z) using SimulLens code:
nside = 8192
npix = 12 * nside**2
noct = npix / 8 - nside
zs=57

input_dir = input.baseDirectory
mapsNames = sorted(directory_contents)[58:1:-1] # use this one when there is a README file in the directory.
#mapsNames = sorted(directory_contents)[57:0:-1]

mapsNames = np.array(mapsNames)
#print("Found these maps in the directory:", mapsNames)
#print("For a total of ",np.shape(mapsNames), "Maps")
#print("*****")
nl = np.shape(mapsNames)[0]
Norm = 1.0

for tomo in range(1,6):

    print('*** Working on tomo' , tomo, '***')

    #----
    # Read the n(z) files
    zs,Nz = np.loadtxt("./Euclid_dndz_fu08_bin1-5.dat",comments='#',usecols=(0,tomo),unpack=True)
    #Nz_normal = Nz/total_area(zs,Nz)
    nzbins = np.shape(zs)[0]

    #----
    # compute the Jacobian, dchi/dz:
    chi_file = np.zeros(nzbins)
    for i in range(0,nzbins):
        chi_file[i] = ncosmo.comoving_distance(zs[i]).value*ncosmo.h
        #print(i,zs[i], chi_file[i])

    #----
    # compute the dchi:
    dchi = np.zeros(nzbins)
    for i in range(0,nzbins-1):
        dchi[i] = chi_file[i+1]-chi_file[i]

    dchi[nzbins-1] = dchi[nzbins-2]


    # Initialise kappa maps:
    kpmap = np.zeros(npix)

    #Loop over the lenses:
    zid=0
    for filename in mapsNames:
        #tmp = filename.split('_')
        #zid = int(tmp[4][:-4])
        #zl = zl_list[zid]
        #zl=dllen[zid, 0]
        #zl=zlens_exact[28-zid]
        zl=zlens_exact[56-zid]
        chi_l = ncosmo.comoving_distance(zl).value*ncosmo.h

        zid+=1

        #find minimum for the Nz integration
        zmin = 0
        for ii in range(0,nzbins):
            if(zs[ii] > zl):
                zmin = ii
                break

        # Get integal over sources
        gw = 0
        for ii in range(zmin, nzbins):
            gw = gw + Nz[ii]*(1.0 - chi_l/chi_file[ii])*dchi[ii]     


        #-----
        # Get Kernel
        kernel = 3./2. * Om0 * (3000/10240)/(3e3**2)*chi_l * (1+zl) * gw * Norm
        #print('Kernel4 = ', kernel)

        #print(zl, zmin, zs[zmin], Nz[zmin], gw, kernel, filename)

        # Get particle_count planes:
        massmap = np.fromfile(input.baseDirectory+filename, dtype="<f")

        #Convert to mass planes, in M_sun/h
        #massmap *= mp 

        #Convert to overdensity maps, 
        massmap-=np.mean(massmap)*8

        # Integrate:
        #kpmap = kpmap + massmap*kernel

    continue    

    # Convert to ring-ordered for sph.harmonic transforms:
    kpmap_ring = hp.reorder(kpmap, n2r = True)
    print("reordered map")

    #Apply mask
    if 'hpmap_weight' in locals():
        print("Mask already defined!")
    else:
        print("Reading in weight map for Mira Titan (Comment if already on disk...)")
        hpmap_weight = hp.fitsfunc.read_map('../../FullSky_weight.fits', field=0,nest=False, partial=False, hdu=1, h=False, verbose=True, memmap=True)


    kpmap_ring *= hpmap_weight
    print("applied mask")


    # show kappa:
    hp.mollview(kpmap_ring,
                unit='kappa', cmap=pl.cm.coolwarm,
                cbar=True, title='', norm=None
                , nest=False
                )
    pl.show()

    lonra = [0, 20]
    latra = [0, 10]

    hp.cartview(kpmap_ring,
                cbar=True, lonra=lonra, latra=latra,
                nest=False,
                title=" ",
                # min=-10, max=0,
                cmap=pl.cm.coolwarm,
                norm=None, unit='kappa')
    pl.show()


    fname = "../../kappa/V0/kappa_tomo"+np.str(tomo)+"_V3" 

    np.save(fname, kpmap_ring)
    print("Saved kappa")




# In[ ]:


# Compute C_ell:
Cl = hp.sphtfunc.anafast(kpmap_ring, map2=None, nspec=None, lmax=5000, mmax=None, iter=1, alm=False, pol=False, use_weights=False, datapath=None, gal_cut=0, use_pixel_weights=False)
Cl *= 8.0
print("Got C_ell")
#fname = "../../kappa/V0/Cl_tomo"+np.str(tomo)+".dat"
fname = "../../kappa/V0/Cl_tomo"+np.str(tomo)+"_V3.dat"
np.savetxt(fname,Cl)
print("Saved C_ell")



# In[ ]:


input_dir = input.baseDirectory
mapsNames = sorted(directory_contents)[58:1:-1] # use this one when there is a README file in the directory.
#mapsNames = sorted(directory_contents)[57:0:-1]

mapsNames = np.array(mapsNames)

zid=0
for filename in mapsNames:
    #tmp = filename.split('_')
    #zid = int(tmp[4][:-4])
    #zl = zl_list[zid]
    #zl=dllen[zid, 0]
    #zl=zlens_exact[28-zid]
    zl=zlens_exact[56-zid]
    chi_l = ncosmo.comoving_distance(zl).value*ncosmo.h

    zid+=1

    #find minimum for the Nz integration
    zmin = 0
    for ii in range(0,nzbins):
        if(zs[ii] > zl):
            zmin = ii
            break

    # Get integal over sources
    gw = 0
    for ii in range(zmin, nzbins):
        gw = gw + Nz[ii]*(1.0 - chi_l/chi_file[ii])*dchi[ii]     

    print(zl, zmin, zs[zmin], Nz[zmin], gw)


# In[ ]:





# In[31]:


print(path_in+filename)


# In[32]:


get_ipython().system('ls -l /global/cfs/cdirs/m5099/FS2/counts_200_1024.npy')


# In[ ]:




