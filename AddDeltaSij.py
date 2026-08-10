#!/usr/bin/env python
# coding: utf-8

# In[4]:


# This code reads in a lensing galaxy catalogue and appends to it 
# values of the mass over-density and tidal field tensor at the galaxy locations
# By Joachim Harnois-Deraps, 2021.

import numpy as np
import healpy as hp
#from scipy.interpolate import interp1d
import random
import input
from pathlib import Path


# The tomographic bin
tomobin = 1

print("**************************")
print("Processing tomo bin",tomobin)
print("**************************")

# Use the normal or smoothed tensor fields: '_V2' (0.1Mpc) or '_smooth' (0.5Mpc) or '_smooth_1Mpc' (1Mpc)
smooth_flag='_smooth_05Mpc'

# the number of galaxies per square arcminute to extract
gpam = 0.6

#SkySim5000:
#zfile=np.loadtxt("z2ts.txt",delimiter=',')
#snaplist = np.flip(zfile[1:58,1].astype(int))


#HACCY1:
nshells=51
zfile=np.loadtxt("z2ts_HACCY1_"+str(nshells)+".txt")
snaplist = np.flip(zfile[:,1].astype(int))
shell_array = np.load('HACC_51shells.dat.npy')
delta_chi = shell_array['chimax'] - shell_array['chimin']

#BCM:
BCM_model='dmo'


n_slices = np.size(snaplist)


# In[2]:


#for plane in range(0,41):
for plane in range(0,n_slices):

    #d_smooth = '1' #'01', '05' or '1'
    d_smooth = '0.5' #'01', '05' or '1'

    print('starting analysis of plane %i' % plane)

    #SkySim5000:
    #V0
    #fname = '../../GalCat/StageIV_nz/GalCat_tomo'+np.str(tomobin)+'_plane'+np.str(plane)+'_'+np.str(gpam)+'GpAM_RA_Dec_g1_g2_w.asc'
    #fname = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo'+np.str(tomobin)+'_plane'+np.str(plane)+'_'+np.str(gpam)+'GpAM_RA_Dec_g1_g2_w_z.asc'

    #V1
    #fname = '../../GalCat/StageIV_nz/V1/GalCat_tomo'+np.str(tomobin)+'_plane'+np.str(plane)+'_'+np.str(gpam)+'GpAM_RA_Dec_g1_g2_w_linear_bias.asc'
    #fname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+np.str(tomobin)+'_plane'+np.str(plane)+'_'+np.str(gpam)+'GpAM_RA_Dec_g1_g2_w_linear_bias_bta1.asc'
    #fname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+np.str(tomobin)+'_plane'+np.str(plane)+'_'+np.str(gpam)+'GpAM_RA_Dec_g1_g2_w_linear_bias_bta2.asc'
    #fname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_linear_bias_smooth'+d_smooth+'_bta2.asc'
    # note : the *planes* named bta1 are now bta2.

    #HACC/BCM:
    fname = '../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_nshells51_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z.asc'
    #fname = '../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_nshells51_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_linear_bias.asc'


    # check file exists:
    my_file = Path(fname)
    if my_file.is_file():
        # file exists
        print("Got file! Reading",fname)
        ra, dec, shear1, shear2, w, z = np.loadtxt(fname, unpack=True)

        #load mass maps and tidal tensor
        #SkySim5000:
        #fname = "../../density/density_map_"+str(snaplist[plane])+"_dens_allsky.npy"
        #fname = input.deltaDir+"density_map_"+str(snaplist[plane])+"_dens_allsky_smooth.npy"

        #HACC/BCM:
        fname = f"../../../BCM/{BCM_model}/fullsky_den/density_map_{plane:02d}_dens_allsky.npy"

        print("opening ", fname) 
        hpmap_density = np.load(fname)
        print("Got density file!")

        #SkySim5000:
        #fname = "../../tidalfield/tidal_field_map_"+str(snaplist[plane])+"_allsky"+smooth_flag+".npy"

        #HACC/BCM:
        fname = "../../../BCM/"+BCM_model+"/tidalfield/tidal_field_map_"+str(plane)+"_allsky"+smooth_flag+".npy"

        print("opening ", fname) 
        hpmap_tidal = np.load(fname)
        print("Got tidal file!")


        # Normalize the maps correctly:       
        mean_rho = (np.mean(hpmap_density)) # no factor of 8.0 since we have full sky data here
        hpmap_density /= mean_rho 
        hpmap_density -= 1

        # should express this in terms of shell thickness...
        # SkySim5000 Normalisation:
        #Norm_empirical = 0.6525 

        #HACC Normalisation:
        #Norm_empirical = 0.6525
        Norm_empirical =0.6526*(114/delta_chi[plane])**(-0.5)


        hpmap_tidal *= Norm_empirical

        print("over density:",np.min(hpmap_density), np.max(hpmap_density), np.mean(hpmap_density))
        print("s11:",np.min(hpmap_tidal[:,0]), np.max(hpmap_tidal[:,0]), np.mean(hpmap_tidal[:,0]))
        print("s22:",np.min(hpmap_tidal[:,1]), np.max(hpmap_tidal[:,1]), np.mean(hpmap_tidal[:,1]))
        print("s12:",np.min(hpmap_tidal[:,2]), np.max(hpmap_tidal[:,2]), np.mean(hpmap_tidal[:,2]))



        # Interpolate maps at galaxy positions
        den_pix = hp.get_interp_val(hpmap_density, (90.-dec)*np.pi/180., ra*np.pi/180., nest=False )

        s11_pix = hp.get_interp_val(hpmap_tidal[:,0], (90.-dec)*np.pi/180., ra*np.pi/180., nest=False )
        s22_pix = hp.get_interp_val(hpmap_tidal[:,1], (90.-dec)*np.pi/180., ra*np.pi/180., nest=False )
        s12_pix = hp.get_interp_val(hpmap_tidal[:,2], (90.-dec)*np.pi/180., ra*np.pi/180., nest=False )



        # Save a shear catalogue in standard format to be read by TreeCorr
        #SkySim:
        #outname = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij'+smooth_flag+'.asc'
        #outname = input.GalDir+'StageIV_nz/V0/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_den_Sij.asc'
        #outname = '../../GalCat/StageIV_nz/V1/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias'+smooth_flag+'.asc'
        #outname = '../../GalCat/StageIV_nz/V1/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias'+smooth_flag+'_bta2.asc'
        #outname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias'+smooth_flag+'_bta1.asc'
        #outname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias'+smooth_flag+'_bta2.asc'
        #outname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth'+d_smooth+'_'+smooth_flag+'_bta1.asc'
        #outname = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth'+d_smooth+'_'+smooth_flag+'_bta2_new.asc'

        #HACC/BCM:
        outname = '../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_nshells51_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_'+smooth_flag+'.asc'
        #outname = '../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(tomobin)+'_plane'+str(plane)+'_nshells51_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_'+smooth_flag+'_linear_bias.asc'

        print("Wrinting",outname)

        # given ra=np.degrees(phi), e2 DOES NOT NEED FLIPPING.
        #np.savetxt(outname, np.c_[ra_rand, dec_rand, shear1_pix_front*dz1 + shear1_pix_back*dz2, shear2_pix_front*dz1 + shear2_pix_back*dz2, np.ones(ngal_cur[plane]),spec_z[gal_cur],kappa_pix_front*dz1 + kappa_pix_back*dz2 ])
        np.savetxt(outname, np.c_[ra, dec, shear1, shear2, w,z, den_pix, s11_pix, s22_pix, s12_pix])



    else:
        print('found no file named '+fname)


print("Done!")


# In[5]:


ls -l  ../../../BCM/dmo/GalCat/GalCat_tomo1_plane*


# In[6]:


import os

print('Concatenating SkySim5000 plane catalogues')
#SkySim5000

#V0:
#os.system("for tomo in {1..1}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/*tomo$tomo*0.6GpAM*; done")
#os.system("for tomo in {1..1}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/*tomo$tomo*0.6GpAM* > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z.asc'; done")
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/*tomo$tomo*0.6GpAM*_V2.asc; done")
#os.system("for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/*tomo$tomo*0.6GpAM*_V2.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.1Mpc.asc'; done")
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/*tomo$tomo*0.6GpAM*_smooth.asc; done")
#os.system("for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/*tomo$tomo*0.6GpAM*_smooth.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc'; done")

#V1:
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/*tomo$tomo*0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias.asc; done")
#os.system("for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/*tomo$tomo*0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo$tomo'_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias.asc'; done")
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/*tomo$tomo*0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc; done")
#os.system("for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/*tomo$tomo*0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo$tomo'_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'; done")
#os.system("for tomo in {2..2}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*tomo$tomo*0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_1Mpc.asc; done")
#os.system("for tomo in {2..2}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*tomo$tomo*0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_1Mpc.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo$tomo'_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_1Mpc.asc'; done")
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; rm -v   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*All_0.6GpAM*; done")
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc; done")
#os.system("for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'; done")
#os.system("for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta2.asc; done")
#os.system("for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta2.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta2.asc'; done")
#os.system("let bta=2; for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta$bta.asc; done")
#os.system("let bta=2; for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta$bta.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta'$bta'_smooth0.5Mpc.asc'; done")
#os.system("let bta=2; for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_V2_bta$bta.asc; done")
#os.system("let bta=2; for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_V2_bta$bta.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta'$bta'_smooth0.1Mpc.asc'; done")
#os.system("let bta=1; smooth='1';for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth$smooth'__smooth_bta'$bta.asc; done")
#os.system("let bta=1; smooth='1';for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth$smooth'__smooth_bta'$bta.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth'$smooth'_bta'$bta'_smooth0.5Mpc.asc'; done")
#os.system("let bta=2; smooth='1';for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth$smooth'__smooth_bta'$bta.asc; done")
#os.system("let bta=2; smooth='1';for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth$smooth'__smooth_bta'$bta.asc > /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth'$smooth'_bta'$bta'_smooth0.5Mpc.asc'; done")

#HACK/BCM:
os.system("BCM='dmo'; for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij__smooth_05Mpc.asc; done")
os.system("BCM='dmo'; for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij__smooth_05Mpc.asc > /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc'; done")
#os.system("BCM='dmo'; for tomo in {1..5}; do echo 'working on tomo'$tomo; ls -l   /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij__smooth_05Mpc_linear_bias.asc; done")
#os.system("BCM='dmo'; for tomo in {1..5}; do cat   /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/*tomo$tomo*0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij__smooth_05Mpc_linear_bias.asc > /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/GalCat_tomo$tomo'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc_linear_bias.asc'; done")


# In[21]:


# Clean up
#HACK/BCM:
os.system("BCM='dmb'; rm -v  /global/homes/j/jharno/IA-infusion/BCM/$BCM/GalCat/*plane*")


# In[ ]:


# For visualisation:
import matplotlib.pyplot as plt

plane = 56
Norm_empirical = 0.6525 

smooth_flag='_V2'
fname = "../../tidalfield/tidal_field_map_"+str(snaplist[plane])+"_allsky"+smooth_flag+".npy"

print("opening ", fname) 
hpmap_tidal = np.load(fname)
print("Got tidal file!")
#hpmap_tidal *= Norm_empirical

print("s11:",np.min(hpmap_tidal[:,0]), np.max(hpmap_tidal[:,0]), np.mean(hpmap_tidal[:,0]))
print("s22:",np.min(hpmap_tidal[:,1]), np.max(hpmap_tidal[:,1]), np.mean(hpmap_tidal[:,1]))
print("s12:",np.min(hpmap_tidal[:,2]), np.max(hpmap_tidal[:,2]), np.mean(hpmap_tidal[:,2]))

lonra = [0, 3]
latra = [0, 3]
hp.cartview(np.log(hpmap_tidal[:,0]), cbar=True, lonra=lonra, latra=latra,
        nest=False,
            title="plane "+np.str(plane),
            # min=-10, max=0,
            cmap=plt.cm.coolwarm,            
            norm=None, unit='$t_{11}$')
plt.show()

#--------------------
smooth_flag='_smooth'
fname = "../../tidalfield/tidal_field_map_"+str(snaplist[plane])+"_allsky"+smooth_flag+".npy"

print("opening ", fname) 
hpmap_tidal2 = np.load(fname)
print("Got tidal file!")
#hpmap_tidal *= Norm_empirical

print("s11:",np.min(hpmap_tidal2[:,0]), np.max(hpmap_tidal2[:,0]), np.mean(hpmap_tidal2[:,0]))
print("s22:",np.min(hpmap_tidal2[:,1]), np.max(hpmap_tidal2[:,1]), np.mean(hpmap_tidal2[:,1]))
print("s12:",np.min(hpmap_tidal2[:,2]), np.max(hpmap_tidal2[:,2]), np.mean(hpmap_tidal2[:,2]))

lonra = [0, 3]
latra = [0, 3]
hp.cartview(np.log(hpmap_tidal2[:,0]), cbar=True, lonra=lonra, latra=latra,
        nest=False,
            title="plane "+np.str(plane),
            # min=-10, max=0,
            cmap=plt.cm.coolwarm,            
            norm=None, unit='$t_{11}$')
plt.show()


# In[2]:





# In[27]:


import matplotlib.pyplot as plt
GalCat = np.loadtxt('../../../BCM/dmb/GalCat/GalCat_tomo1_All_nshells51_0.6GpAM_RA_Dec_g1_g2_w_z_linear_bias.asc')
hp.mollview()
hp.projscatter(GalCat[:,0],GalCat[:,1], lonlat=True)
plt.show()


# In[9]:


# Clean up catalogues for boundary effects, where shear is sometimes -1e30...
# Must run after concatenating the plane catalogues
HACCY1=1
GpAM = 0.6
seed=0
nshells=51
if(HACCY1):
    #path='/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/StageIV_nz/HACC150/V0/seed'+str(seed)+'/'
    #path='/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_2/noIA/seed'+str(seed)+'/'
    #path='/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed'+str(seed)+'/'
    #path='/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150/noIA/seed'+str(seed)+'/'

    #BCM:
    path='/global/homes/j/jharno/IA-infusion/BCM/dmo/GalCat/'


    for tomo in range(1,2):
        #fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z.asc'
        #fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_good.asc'
        #fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_nosubsample.asc'
        #fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_good_nosubsample.asc'
        #fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_zclmax100.asc'
        #fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_good_clmax100.asc'
        #fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_subsampledensitymax_nside8192.asc'
        #fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192.asc'
        #fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_nosubsample_nside8192.asc'
        #fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_good_nosubsample_nside8192.asc'
        #fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192_linear_bias.asc'
        #fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(gpam)+'GpAM_RA_Dec_g1_g2_w_z_verygood_subsampledensitymax_nside8192_linear_bias.asc'

        #BCM:
        fname_in='GalCat_tomo'+str(tomo)+'_All_'+str(GpAM)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc'
        fname_out='GalCat_tomo'+str(tomo)+'_All_'+str(GpAM)+'GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc_good.asc'


        GalCat = np.loadtxt(path+fname_in)
        Good_cat = GalCat[GalCat[:,3]<10]
        np.savetxt(path+fname_out,Good_cat)
        Bad_cat = GalCat[GalCat[:,3]>1]
        print("Removed "+str(np.shape(Bad_cat)[0])+" bad galaxies")
        print('Cleaned GalCat for tomo',tomo)


# In[ ]:


#BCM:
fname_in='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(GpAM)+'GpAM_RA_Dec_g1_g2_w_z.asc'
fname_out='GalCat_tomo'+str(tomo)+'_All_nshells'+str(nshells)+'_'+str(GpAM)+'GpAM_RA_Dec_g1_g2_w_z_good.asc'

GalCat = np.loadtxt(path+fname_in)
Good_cat = GalCat[GalCat[:,3]>-1]
veryGood_cat = Good_cat[Good_cat[:,3]<10]
np.savetxt(path+fname_out,veryGood_cat)

Bad_cat = GalCat[GalCat[:,3]<-1]
veryBad_cat = Bad_cat[Bad_cat[:,3]>10]

print("Removed "+str(np.shape(veryBad_cat)[0])+" bad galaxies")
print('Cleaned GalCat for tomo',tomo)


# In[5]:


71/114


# In[6]:


71/delta_chi


# In[7]:


np.mean(delta_chi)


# In[8]:


71/145


# In[ ]:




