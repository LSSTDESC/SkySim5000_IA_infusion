#!/usr/bin/env python
# coding: utf-8

# In[2]:


import treecorr
import pyccl as ccl
import numpy as np    
from astropy.table import Table

#SkySim5000:
config_shear = {'ra_col' : 1,
          'dec_col' : 2,
          'g1_col' : 4,
          'g2_col' : 5,
          'w_col' : 6,
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

# HACC:
config_shear_HACC = {'ra_col' : 1,
          'dec_col' : 2,
          'g1_col' : 4,
          'g2_col' : 5,
          'w_col' : 6,
          'dec_units' : 'deg',
          'ra_units' : 'deg'}

config_shear_HOD = {'ra_col' : 1,
          'dec_col' : 2,
          'g1_col' : 4,
          'g2_col' : 5,
          'w_col' : 6,
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

config_IA_noisy = {'ra_col' : 1,
          'dec_col' : 2,
          'g1_col' : 8,
          'g2_col' : 9,
          'w_col' : 6,
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

config_IA = {'ra_col' : 1,
          'dec_col' : 2,
          'g1_col' : 10,
          'g2_col' : 11,
          'w_col' : 6,
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

config_pureIA = {'ra_col' : 1,
          'dec_col' : 2,
          'g1_col' : 12,
          'g2_col' : 13,
          'w_col' : 6,
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

config_fits = {'ra_col' : 'ra',
          'dec_col' : 'dec',
          'g1_col' : 'gamma1',
          'g2_col' : 'gamma2',
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

config_lenses = {'ra_col' : 1,
          'dec_col' : 2,
          'w_col' : 6,
          'ra_units' : 'deg',
          'dec_units' : 'deg'}

# BCM
#BCM_model='dmo'
#BCM_model='dmb'
BCM_model='dmb_theta_ej_5'

#smooth_flag=''
smooth_flag='_0.5Mpc'
#smooth_flag='_smooth0.5Mpc'

smooth_den = '_smooth1'


# We load all of the data into a single array for simplicity, n=6 is used for consistency so that we can start counting from 1 to 5 instead of from 0 to 4
# 
# Here we calculate the auto correlation functions for all bins

# In[2]:


nside=8192
n_shells=51
seed=0


# Pollux:

dndz_flag='SRDv1_nz'
kappa_dir=dndz_flag
#folder_path='/pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
folder_path='/pscratch/sd/j/jmena/HOS_Sims_baryonified/'+BCM_model+'/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir

#BCM_model='dmo'
#smooth_flag=''

filename='kappa_hacc_seed'+str(seed)+'_nside'+str(nside)+'_'
downsample=False
from astropy.io import fits
import random


nshells=51
#for i in range(4,6):
for i in range(1,6):
    for j in range(1, i+1):
    #for j in range(4, 5):
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_clmax100.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_clmax100.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_nosubsample.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_nosubsample.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_nosubsample_nside8192.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_nosubsample_nside8192.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed1/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed1/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192.asc'.format(j,nshells)

        #HOD:
        filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM.dat'.format(i)
        filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM.dat'.format(j)

        #HACC + IA
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z.asc'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z.asc'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_linear_bias.asc'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_linear_bias.asc'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/StageIV_nz/V0/seed0/GalCat_tomo{}_All_nshells{}_0.06GpAM_RA_Dec_g1_g2_w_z_good.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/StageIV_nz/V0/seed0/GalCat_tomo{}_All_nshells{}_0.06GpAM_RA_Dec_g1_g2_w_z_good.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/StageIV_nz/V0/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/StageIV_nz/V0/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_verygood_subsampledensitymax_nside8192_linear_bias.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_verygood_subsampledensitymax_nside8192_linear_bias.asc'.format(j,nshells)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192_linear_bias_bta2.asc'.format(i,nshells)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/GalCat_tomo{}_All_nshells{}_0.6GpAM_RA_Dec_g1_g2_w_z_good_subsampledensitymax_nside8192_linear_bias_bta2.asc'.format(j,nshells)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}.asc'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}_m1.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}_m1.asc'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_'+smooth_den+'bta1_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_'+smooth_den+'bta1_smooth{}.asc'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta2_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta2_smooth{}.asc'.format(j,smooth_flag)

        #BCM:
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/GalCat_tomo{}_All_nshells51_0.6GpAM_RA_Dec_g1_g2_w_z.asc'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/GalCat_tomo{}_All_nshells51_0.6GpAM_RA_Dec_g1_g2_w_z.asc'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/GalCat_tomo{}_All_nshells51_0.6GpAM_RA_Dec_g1_g2_w_z.asc'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/GalCat_tomo{}_All_nshells51_0.6GpAM_RA_Dec_g1_g2_w_z.asc'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo'+str(i)+smooth_flag+'.dat'
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo'+str(j)+smooth_flag+'.dat'
        #cat_1 = treecorr.Catalog(filename1, config_shear, flip_g2='True',first_row=2)
        #cat_2 = treecorr.Catalog(filename2, config_shear, flip_g2='True',first_row=2)
        #cat_1 = treecorr.Catalog(filename1, config_shear_HACC, flip_g2='True',first_row=2)
        #cat_2 = treecorr.Catalog(filename2, config_shear_HACC, flip_g2='True',first_row=2)


        # For Juan's catalogue
        #filename1 = folder_path+'/'+filename+'catpoisson_imap'+str(i-1)+'.fits'
        #filename2 = folder_path+'/'+filename+'catpoisson_imap'+str(j-1)+'.fits'
        #filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1_imap'+str(i-1)+'.fits'
        #filename2 = folder_path+'/'+filename+'catpoisson_interppixels_b1_imap'+str(j-1)+'.fits'
        #kappa_hacc_seed3_nside8192_catpoisson_interppixels_b1_imap0.fits
#         filename1 = folder_path+'/'+filename+'catuniform_imap'+str(i-1)+'.fits'
#         filename2 = folder_path+'/'+filename+'catuniform_imap'+str(j-1)+'.fits'

        if(downsample):
             ngal = 0.6 #desired ngal per arcmin sq.
             hdul1=fits.open(filename1)
             hdul2=fits.open(filename2)

             data1=hdul1[1].data
             header1=hdul1[1].header

             data2=hdul2[1].data
             header2=hdul2[1].header
             data1_dict={}
             data2_dict={}
             for column in data1.columns:
                 if column.name in ['ra','dec','kappa','gamma1','gamma2']:
                     data1_dict[column.name]=data1[column.name]
                     data2_dict[column.name]=data2[column.name]

             # Let's downsample to 0.6 gal/arcmin**2

             selection1=sorted(random.sample(range(len(data1_dict['ra'])),int(0.6*(4*np.pi*(180/np.pi)**2/8*60**2))))
             selection2=sorted(random.sample(range(len(data2_dict['ra'])),int(0.6*(4*np.pi*(180/np.pi)**2/8*60**2))))

             data1_dict_downsampled={}
             data2_dict_downsampled={}
             for column in data1_dict.keys():
                 data1_dict_downsampled[column]=data1_dict[column][selection1]
                 data2_dict_downsampled[column]=data2_dict[column][selection2]

             print(len(data1_dict_downsampled['ra'])/(4*np.pi*(180/np.pi)**2/8*60**2)) # this should print 0.6
             print(len(data1_dict_downsampled['ra'])/(4*np.pi*(180/np.pi)**2/8*60**2)) # this should print 0.6
             cat_1 = treecorr.Catalog(ra=data1_dict_downsampled['ra'], dec=data1_dict_downsampled['dec'], g1=data1_dict_downsampled['gamma1'], g2=data1_dict_downsampled['gamma2'], ra_units='deg',dec_units='deg',  flip_g2='True')
             cat_2 = treecorr.Catalog(ra=data2_dict_downsampled['ra'], dec=data2_dict_downsampled['dec'], g1=data2_dict_downsampled['gamma1'], g2=data2_dict_downsampled['gamma2'], ra_units='deg',dec_units='deg',  flip_g2='True')

        else:

             #SkySim JHD::
             #cat_1 = treecorr.Catalog(filename1, config_fits, flip_g2='True')
             #cat_2 = treecorr.Catalog(filename2, config_fits, flip_g2='True')

             #HACC:
             #cat_1 = treecorr.Catalog(filename1, config_shear_HACC, flip_g2='True',first_row=2)
             #cat_2 = treecorr.Catalog(filename2, config_shear_HACC, flip_g2='True',first_row=2)

             #For HOD:
             cat_1 = treecorr.Catalog(filename1, config_shear_HOD, flip_g2='False')
             cat_2 = treecorr.Catalog(filename2, config_shear_HOD, flip_g2='False')

        #gg = treecorr.GGCorrelation(nbins = 20, min_sep = 0.005, max_sep = 7, bin_slop=0.01) 
        gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.01)
        #gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.05)
        gg.process(cat_1, cat_2)
        #gg.process(cat_1, cat_2, metric = 'Arc')
        #gg.write('correlations/cross_correlation_tomo{}_{}'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/StageIV_nz/V0/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/correlations/cross_correlation_tomo{}_{}'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V0/correlations/cross_correlation_0.06GpAM_tomo{}_{}'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/correlations/cross_correlation_0.06GpAM_tomo{}_{}_linear_bias_smooth'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/correlations/cross_correlation_0.6GpAM_tomo{}_{}_linear_bias_smooth'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/V1/correlations/cross_correlation_0.06GpAM_tomo{}_{}_linear_bias'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/cross_correlation_0.6GpAM_tomo{}_{}_smooth{}'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/cross_correlation_0.6GpAM_tomo{}_{}_smooth{}_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/cross_correlation_0.6GpAM_tomo{}_{}_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/cross_correlation_0.6GpAM_tomo{}_{}_smooth1_smooth{}_bta2'.format(j,i,smooth_flag))

        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_2/noIA/seed1/correlations/cross_correlation_tomo{}_{}_nshells{}'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_clmax100'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_nosubsample'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_nosubsample_nside8192'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed1/correlations/cross_correlation_tomo{}_{}_nshells{}_subsampledensitymax_nside8192'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_subsampledensitymax_nside8192'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_subsampledensitymax_nside8192_linearbias'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_subsampledensitymax_nside8192_linearbias_bta2'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_subsampledensitymax_nside8192_linearbias_bta2_juan'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_nshells{}_subsampledensitymax_nside8192_uniform_juan'.format(j,i,nshells))

        #HACCY1+IA
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_poisson_noIA'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_noIA{}_pollux'.format(j,i,smooth_flag))


        #BCM
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_noIA_new'.format(j,i,nshells))
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_noIA_linear_bias'.format(j,i,nshells))


        # For HOD:
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/cross_correlation_tomo{}_{}'.format(j,i))
        gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/cross_correlation_tomo{}_{}_accurate'.format(j,i))


        print("Wrote treecorr for bins ",i,j)


# Here we show how to read the correlation functions saved to a file by TreeCorr

# In[3]:


# Run the auto and cross-correlation functions on the mocks with IA infused, no noise:

# And the cross-redshift bins
for i in range(5,6):
    for j in range(3, 4):
# for i in range(1,6):
#    for j in range(1, i+1):
        #filename1 = '/global/cscratch1/sd/lhior/fin_epsilon_IA_{}.dat'.format(i)
        #filename2 = '/global/cscratch1/sd/lhior/fin_epsilon_IA_{}.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2corr.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2corr.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_smooth_V2corr.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_smooth_V2corr.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_bta2.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_bta2.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_NLA.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_NLA.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_V3.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_V3.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_nodelta.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_nodelta.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(j)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/NLA/galcat_skysim5000_AIAp1_tomo{}_nofactor2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/NLA/galcat_skysim5000_AIAp1_tomo{}_nofactor2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_m1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/TT/galcat_skysim5000_c2p1_tomo{}_nofactor2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/TT/galcat_skysim5000_c2p1_tomo{}_nofactor2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT_m1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2_m1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_m1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta1_to_bta0.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta1_to_bta0.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_TT.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_AIAp1_bta1_tomo{}_nofactor2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_AIAp1_bta1_tomo{}_nofactor2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/deltaTT/galcat_skysim5000_C2p1_bta1_tomo{}_V2_nofactor2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/deltaTT/galcat_skysim5000_C2p1_bta1_tomo{}_V2_nofactor2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_to_bta0.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_to_bta0.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT_V3.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT_V3.dat'.format(j,smooth_flag)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_V2.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_V2.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_AIA2.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_AIA2.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_nofactor2.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_nofactor2.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_nofactor2.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_nofactor2.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(j)
        filename1='/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo{}{}.dat'.format(i,smooth_flag)        
        filename2='/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo{}{}.dat'.format(j,smooth_flag)        

        cat_1 = treecorr.Catalog(filename1, config_IA, flip_g2='True',first_row=2)
        cat_2 = treecorr.Catalog(filename2, config_IA, flip_g2='True',first_row=2)



        #For HOD:
        #cat_1 = treecorr.Catalog(filename1, config_IA, flip_g2='False',first_row=2)
        #cat_2 = treecorr.Catalog(filename2, config_IA, flip_g2='False',first_row=2)

        #gg = treecorr.GGCorrelation(nbins = 20, min_sep = 0.005, max_sep = 7, bin_slop=0.01) 
        #gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.01)
        gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.05)
        gg.process(cat_1, cat_2)
        #gg.process(cat_1, cat_2, metric = 'Arc')
        #gg.write('correlations/infused_cross_correlation_no_noise_tomo{}_{}'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_TT'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_TT_V2'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_TT_V2corr'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_TT_smooth_V2corr'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_linear_bias'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_linear_bias_smooth'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.06GpAM_linear_bias_smooth_1Mpc'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_linear_bias_smooth'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_linear_bias_smooth_bta2'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_linear_bias_smooth_TT_V3'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_linear_bias_smooth_TT_nodelta'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/NLA/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/TT/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_TT_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/deltaNLA/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta1_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/deltaTT/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta1_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/deltaTT/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta1_V2_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta2_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta1_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_bta1_to_bta0'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth{}_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth05_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth1_smooth{}_bta2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth1_smooth{}_bta1_to_bta0'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth05_smooth{}_bta1_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_smooth05_smooth{}_bta1_TT_V3'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_V2'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_nofactor2'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_TT_nofactor2'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_tomo{}_{}_no_noise_0.6GpAM_TT'.format(j,i))

        #BCM:
        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/seed0/correlations/cross_correlation_tomo{}_{}_poisson_AIAp1'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_AIAp1'.format(j,i))
        gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_AIAp1{}_pollux'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_AIAp1_linear_bias'.format(j,i))


        print('Done tomo'+str(j)+'-'+str(i))


# In[4]:


# Run treecorr on the mocks with IA_TATT infused (II term):
# Auto and cross-redshift bins

for i in range(1,6):
#    for j in range(1, i+1):
#for i in range(1,6):
    for j in range(1, i+1):
        #filename1 = '/global/cscratch1/sd/lhior/fin_epsilon_IA_{}.dat'.format(i)
        #filename2 = '/global/cscratch1/sd/lhior/fin_epsilon_IA_{}.dat'.format(j)
        #V0:
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2.dat'.format(j)

        #filename1 = '../../GalCat/SRD-Y1/NLA/galcat_skysim5000_IA_tomo{}_nofactor2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/NLA/galcat_skysim5000_IA_tomo{}_nofactor2.dat'.format(j,smooth_flag)

        #filename1 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(j)

        #BCM:
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(j)

        #pollux
        filename1='/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo{}{}.dat'.format(i,smooth_flag)        
        filename2='/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo{}{}.dat'.format(j,smooth_flag)        



        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_m1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT_m1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_TT.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT_V3.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT_V3.dat'.format(j,smooth_flag)


        #HOD:
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_AIA2.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_AIA2.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(j)


        #V1:
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_bta2.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_bta2.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_V3.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_V3.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_nodelta.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_nodelta.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(j)
        cat_1 = treecorr.Catalog(filename1, config_pureIA, flip_g2='True',first_row=2)
        cat_2 = treecorr.Catalog(filename2, config_pureIA, flip_g2='True',first_row=2)

        #HOD:
        #cat_1 = treecorr.Catalog(filename1, config_pureIA, flip_g2='False')
        #cat_2 = treecorr.Catalog(filename2, config_pureIA, flip_g2='False')


        #gg = treecorr.GGCorrelation(nbins = 20, min_sep = 0.005, max_sep = 7, bin_slop=0.01) 
        #gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.01)
        gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.05)
        gg.process(cat_1, cat_2)
        #gg.process(cat_1, cat_2, metric = 'Arc')
        #gg.write('correlations/infused_cross_correlation_TATT_tomo{}_{}'.format(j,i))

        #V0:
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.06GpAM'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.06GpAM_TT'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.06GpAM_TT_V2'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/NLA/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_bta2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth{}_TT_m1'.format(j,i,smooth_flag))


        #V1:
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.06GpAM_linear_bias'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.06GpAM_linear_bias_smooth'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_linear_bias_smooth'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_linear_bias_smooth_bta2'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_linear_bias_smooth_TT_V3'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_linear_bias_smooth_TT_nodelta'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.06GpAM_linear_bias_smooth_1Mpc'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth1_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth1_smooth{}_bta1_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_smooth1_smooth{}_bta1_TT_V3'.format(j,i,smooth_flag))

        #gg.write('/global/homes/j/jharno/IA-infusion/HACC-Y1/GalCat/SRD-Y1/HACC150_3/noIA/seed0/correlations/cross_correlation_tomo{}_{}_poisson_II'.format(j,i,nshells))

        #BCM
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_NLA_II'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_deltaNLA_II'.format(j,i))
        gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlation_tomo{}_{}_deltaNLA_II{}_pollux'.format(j,i,smooth_flag))


        #HOD
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations//infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations//infused_cross_correlation_TATT_tomo{}_{}_0.6GpAM_TT'.format(j,i,smooth_flag))


        print('Done tomo'+str(j)+'-'+str(i))


# In[ ]:


# Run treecorr on the mocks with IA_TATT infused (GI term):

# Run auto and cross-redshift bins at once
for i in range(1,6):
    #for j in range(1, i+1):
    for j in range(1, i+1):

        #V0
        #filename1 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2.dat'.format(j)

        #filename1 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCathttps://jupyter.nersc.gov/user/jharno/perlmutter-login-node-base/lab/tree/global/homes/j/jharno/IA-infusion/SkySim5000/tools/SkySim5000_IA_in        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth{}_TT_m1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta1_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta2_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_bta2.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta1_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth{}_TT.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth05_bta1_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth05_smooth{}_bta1.dat'.format(j,smooth_flag)
        #filename1 = '../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta1_smooth{}.asc'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT.dat'.format(j,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth{}_bta1_TT_V3.dat'.format(j,smooth_flag)

        #filename1 = '../../GalCat/SRD-Y1/NLA/galcat_skysim5000_IA_tomo{}_nofactor2.dat'.format(i,smooth_flag)
        #filename2 = '../../GalCat/SRD-Y1/NLA/galcat_skysim5000_IA_tomo{}_nofactor2.dat'.format(j,smooth_flag)

        #BCM:
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(j)
        #pollux
        filename1='/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo{}{}.dat'.format(i,smooth_flag)        
        filename2='/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo{}{}.dat'.format(j,smooth_flag)        


        #HOD
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_AIA2.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_AIA2.dat'.format(j)
        #filename1 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(i)
        #filename2 = '/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(j)


        #V1
        #filename1 = '/global/cscratch1/sd/lhior/fin_epsilon_IA_{}.dat'.format(i)
        #filename2 = '/global/cscratch1/sd/lhior/fin_epsilon_IA_{}.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(i)
        #filename1 = '../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta2.asc'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_bta2.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_V3.dat'.format(j)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_nodelta.dat'.format(j)
        #filename1 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(i)
        #filename2 = '../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(j)

        cat_1 = treecorr.Catalog(filename1, config_shear, flip_g2='True',first_row=2)
        cat_2 = treecorr.Catalog(filename2, config_pureIA, flip_g2='True',first_row=2)


        #HOD:
        #cat_1 = treecorr.Catalog(filename1, config_shear_HOD, flip_g2='False')
        #cat_2 = treecorr.Catalog(filename2, config_pureIA, flip_g2='False')

        #gg = treecorr.GGCorrelation(nbins = 20, min_sep = 0.005, max_sep = 7, bin_slop=0.01) 
        #gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.01)
        gg = treecorr.GGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.05)
        gg.process(cat_1, cat_2)



        #gg.process(cat_1, cat_2, metric = 'Arc')
        #gg.write('correlations/infused_cross_correlation_TATT_tomo{}_{}'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_GI_tomo{}_{}_0.06GpAM'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_GI_tomo{}_{}_0.06GpAM_TT'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V0/correlations/infused_cross_correlation_GI_tomo{}_{}_0.06GpAM_TT_V2'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.06GpAM_linear_bias'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.06GpAM_linear_bias_smooth'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_linear_bias_smooth'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_linear_bias_smooth_bta2'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_linear_bias_smooth_TT_V3'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_linear_bias_smooth_TT_nodelta'.format(j,i))
        #gg.write('../../GalCat/StageIV_nz/V1/correlations/infused_cross_correlation_GI_tomo{}_{}_0.06GpAM_linear_bias_smooth_1Mpc'.format(j,i))

        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_TT_m1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/NLA/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_nofactor2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_bta2'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth{}_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth05_smooth{}_bta1'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth1_smooth{}_bta1_TT'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/lin_bias_pos/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_smooth1_smooth{}_bta1_TT_V3'.format(j,i,smooth_flag))

        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlations_tomo{}_{}_NLA_GI'.format(j,i,smooth_flag))
        gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlations_tomo{}_{}_deltaNLA_GI{}_pollux'.format(j,i,smooth_flag))
        #gg.write('/global/homes/j/jharno/IA-infusion/BCM/'+BCM_model+'/GalCat/correlations/cross_correlations_tomo{}_{}_deltaNLA_GI'.format(j,i,smooth_flag))


        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM'.format(j,i))
        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/HOD/FromCharlie/correlations/infused_cross_correlation_GI_tomo{}_{}_0.6GpAM_TT'.format(j,i))


        print('Done tomo'+str(j)+'-'+str(i))


# In[ ]:


#mv -v  /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/random_pos/correlations/*nofactor2 /global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/NLA/correlations/


# In[14]:


#####
#GGL: Run the auto and cross-correlation functions on the mocks without IA infused, no noise:
####

for i in range(5,6):
    #for j in range(3, i+1):
    for j in range(3, 4):

        #filename1 = '../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_AIAp1_bta1_tomo{}.dat'.format(i)
        #filename2 = '../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_AIAp1_bta1_tomo{}.dat'.format(j)
        filename1 = '../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_noIA_bta1_tomo{}.dat'.format(i)
        filename2 = '../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_noIA_bta1_tomo{}.dat'.format(j)

        cat_1 = treecorr.Catalog(filename1, config_shear,flip_g1='True', first_row=2)
        cat_2 = treecorr.Catalog(filename2, config_lenses, first_row=2)

        gg = treecorr.NGCorrelation(nbins = 20, min_sep=0.5, max_sep=475.5, sep_units='arcmin',bin_slop=0.05)
        gg.process(cat_2, cat_1)

        #gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/deltaNLA/correlations/GGL_noisefree_AIAp1_tomo{}_{}_0.6GpAM'.format(j,i))
        gg.write('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/SRD-Y1/deltaNLA/correlations/GGL_noisefree_noIA_tomo{}_{}_0.6GpAM'.format(j,i))


# In[23]:


import matplotlib.pyplot as plt
gt = gg.xi
theta = gg.meanr
plt.loglog(theta, gt)
plt.xlabel(r'$\vartheta {\rm [arcmin]}$',fontsize=16)
plt.ylabel(r'$\gamma_{\rm T}(\vartheta)$',fontsize=16)


# In[22]:


theta


# In[15]:


ls /pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z51_subsampledensitymax1.5_nside8192/SRDv1_nz/kappa_hacc_seed0_nside8192_catpoisson_imap0.fits


# In[ ]:


pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z51_subsampledensitymax1.5_nside8()
192/SRDv1_nz/kappa_hacc_seed0_nside8192_catpoisson_interpixels_b1_imap0.fits


# In[3]:


header2


# In[ ]:


1+1


# In[ ]:




