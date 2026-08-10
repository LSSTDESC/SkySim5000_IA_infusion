#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pylab', 'inline')
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
import pyccl.ccllib as lib
import pyccl as ccl
import pyccl.nl_pt as pt
from astropy.table import Table
import numpy as np


# In[4]:


cosmo = ccl.Cosmology(
    Omega_c=0.22, Omega_b=0.0448, 
    h=0.71, sigma8 = 0.801, n_s= 0.963,w0=-1.00,wa=0.0, Omega_k=0.0)

Om_m = cosmo['Omega_m']
rho_crit = lib.cvar.constants.RHO_CRITICAL
Aia=1.0 #1.5,2
C2=1.0
sigma_epsilon = 0.27


# In[3]:


def Epsilon1_NLA(cosmo,z,A1,rho_crit,sxx,syy):
    gz = ccl.growth_factor(cosmo, 1./(1+z))
    Fact = -1*A1*5e-14*rho_crit*cosmo['Omega_m']/gz
    e1_NLA =  Fact  * (sxx - syy) 
    return e1_NLA

def Epsilon2_NLA(cosmo,z,A1,rho_crit,sxy):
    gz = ccl.growth_factor(cosmo, 1./(1+z))
    Fact = -1*A1*5e-14*rho_crit*cosmo['Omega_m']/gz
    e2_NLA = Fact* sxy
    return e2_NLA

def Epsilon1_TT(cosmo,z,A2,rho_crit,sxx,syy):
    gz = ccl.growth_factor(cosmo, 1./(1+z))
    Fact =  -5*A2*5e-14*rho_crit*cosmo['Omega_m']/gz**2
    e1_TT =  Fact * (sxx**2 - syy**2) 
    return e1_TT

def Epsilon2_TT(cosmo,z,A2,rho_crit,sxx,syy,sxy):
    gz = ccl.growth_factor(cosmo, 1./(1+z))
    Fact =  -5*A2*5e-14*rho_crit*cosmo['Omega_m']/gz**2
    e2_TT =  Fact * sxy*(sxx+syy)
    return e2_TT


# In[4]:


for i in np.arange(2,6):
    #SkySim5000
    #catalog = np.loadtxt('../../GalCat/StageIV_nz/V1/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc')  
    #catalog = np.loadtxt('../../GalCat/StageIV_nz/V1/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta2.asc')  
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/random_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.1Mpc.asc')  
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/random_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc')  
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta1_smooth0.5Mpc.asc')  
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta2_smooth0.5Mpc.asc')  
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta1_smooth0.5Mpc.asc')
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta2_smooth0.5Mpc.asc')

    #HACC/BCM:
    BCM_model='dmo'
    catalog = np.loadtxt('../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc')
    #catalog = np.loadtxt('../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc_linear_bias.asc')

    print('Got Gatalogue')

    s11=catalog[:,7]
    s22=catalog[:,8]
    s12=catalog[:,9]
    z=catalog[:,5]

    e1_NLA=Epsilon1_NLA(cosmo,z,Aia,rho_crit,s11,s22)
    e2_NLA=Epsilon2_NLA(cosmo,z,Aia,rho_crit,s12)

    e1_TT=Epsilon1_TT(cosmo,z,C2,rho_crit,s11,s22)
    e2_TT=Epsilon2_TT(cosmo,z,C2,rho_crit,s11,s22,s12)

    random1=np.random.normal(0, sigma_epsilon, e1_NLA.shape)
    random2=np.random.normal(0, sigma_epsilon, e2_NLA.shape)

    print('Got Shapes')

    epsilon = Table([e1_NLA, e2_NLA, e1_TT, e2_TT, random1, random2], names = ('e1_NLA', 'e2_NLA', 'e1_TT', 'e2_TT', 'e1_noise', 'e2_noise'))

    #SkySim5000:
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.6GpAM_linear_bias_smooth.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.6GpAM_linear_bias_smooth_bta2.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth_1Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.1Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc_nofactor2.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_smooth_bta2_0.5Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_sI can confirm that Juan's Tidal field maps are very similar to mine, mooth1_bta1_smooth05Mpc_nofactor2.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_smooth1_bta2_smooth05Mpc.asc'.format(i), format='ascii', overwrite=True)    

    #HACC/BCM:
    epsilon.write('../../../BCM/'+BCM_model+'/GalCat/epsilon_{}_0.6GpAM_smooth0.5Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../../BCM/'+BCM_model+'/GalCat/epsilon_{}_0.6GpAM_smooth0.5Mpc_linear_bias.asc'.format(i), format='ascii', overwrite=True)    

    print('Done tomo',i)


# In[5]:


ls ../../../BCM/dmo/GalCat/epsilon_*


# In[9]:


# Use this cell instead for HOD galaxy catalogues produced by Charlie
for i in np.arange(1,6):
    z,s11,s12,s22 = np.loadtxt('../../GalCat/HOD/FromCharlie/GalCat_tomo'+str(i)+'_06GpAM_new.dat', usecols = (1,13,14,15), unpack=True)

    print('Got Gatalogue')

    #s11=catalog[:,13]
    #s22=catalog[:,15]
    #s12=catalog[:,14]
    #z=catalog[:,2]

    e1_NLA=Epsilon1_NLA(cosmo,z,Aia,rho_crit,s11,s22)
    e2_NLA=Epsilon2_NLA(cosmo,z,Aia,rho_crit,s12)

    e1_TT=Epsilon1_TT(cosmo,z,C2,rho_crit,s11,s22)
    e2_TT=Epsilon2_TT(cosmo,z,C2,rho_crit,s11,s22,s12)

    random1=np.random.normal(0, sigma_epsilon, e1_NLA.shape)
    random2=np.random.normal(0, sigma_epsilon, e2_NLA.shape)

    print('Got Shapes')

    epsilon = Table([e1_NLA, e2_NLA, e1_TT, e2_TT, random1, random2], names = ('e1_NLA', 'e2_NLA', 'e1_TT', 'e2_TT', 'e1_noise', 'e2_noise'))

    epsilon.write('../../GalCat/HOD/FromCharlie/epsilon_{}_0.6GpAM.asc'.format(i), format='ascii', overwrite=True)    

    print('Done tomo',i)


# In[4]:


# Use this instead for mock catalogues produced by Pollux.
from astropy.io import fits


# HACC cataloguesnow, or 
nside=8192
n_shells=51
seed=0
# dndz_flag='Euclid_dndz_fu08_bin1-5'
dndz_flag='SRDv1_nz'
kappa_dir=dndz_flag

#IA tidal field:
#smooth_flag='' #no smoothing
smooth_flag='_0.5Mpc'

#BCM model:
BCM_model='dmb_theta_ej6'
#BCM_model='dmb_dmo'
#BCM_model='dmb'

#folder_path='/pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
folder_path='/pscratch/sd/j/jmena/HOS_Sims_baryonified/'+BCM_model+'/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
filename='kappa_hacc_seed'+str(seed)+'_nside'+str(nside)+'_'
#BCM_model='dmo'
BCM_model='dmb_theta_ej_6'


nshells=51
for i in range(1,6):
    filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1'+smooth_flag+'_imap'+str(i-1)+'.fits'
    #filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1_0.5Mpc_imap'+str(i-1)+'.fits'
    hdul1=fits.open(filename1)
    data1=hdul1[1].data
    header1=hdul1[1].header


    data1_dict={}
    for column in data1.columns:
        if column.name in ['z','t11'+smooth_flag, 't22'+smooth_flag, 't12'+smooth_flag]:
            data1_dict[column.name]=data1[column.name]

    #if(smooth_flag=='_0.5Mpc'):
    #    print('#### HEAVY HACK! INTERCHANGING t22 and t12 TO FIX BUT IN POLLUX ####')
    #    e1_NLA=Epsilon1_NLA(cosmo,data1_dict['z'],Aia,rho_crit,data1_dict['t11'+smooth_flag],-data1_dict['t12'+smooth_flag])
    #    e2_NLA=Epsilon2_NLA(cosmo,data1_dict['z'],Aia,rho_crit,data1_dict['t22'+smooth_flag])

    #    e1_TT=Epsilon1_TT(cosmo,data1_dict['z'],C2,rho_crit,data1_dict['t11'+smooth_flag],data1_dict['t12'+smooth_flag])
    #    e2_TT=Epsilon2_TT(cosmo,data1_dict['z'],C2,rho_crit,data1_dict['t11'+smooth_flag],data1_dict['t12'+smooth_flag],data1_dict['t22'+smooth_flag])

        #Catch galaxies with weird ellipticities
    #    mask=np.where(e2_NLA==0)
    #    print('Zero IA:')
    #    print(data1_dict['t11'+smooth_flag][mask],data1_dict['t22'+smooth_flag][mask],data1_dict['t12'+smooth_flag][mask])

    #else:
    e1_NLA=Epsilon1_NLA(cosmo,data1_dict['z'],Aia,rho_crit,data1_dict['t11'+smooth_flag],data1_dict['t22'+smooth_flag])
    e2_NLA=Epsilon2_NLA(cosmo,data1_dict['z'],Aia,rho_crit,data1_dict['t12'+smooth_flag])

    e1_TT=Epsilon1_TT(cosmo,data1_dict['z'],C2,rho_crit,data1_dict['t11'+smooth_flag],data1_dict['t22'+smooth_flag])
    e2_TT=Epsilon2_TT(cosmo,data1_dict['z'],C2,rho_crit,data1_dict['t11'+smooth_flag],data1_dict['t22'+smooth_flag],data1_dict['t12'+smooth_flag])

    random1=np.random.normal(0, sigma_epsilon, e1_NLA.shape)
    random2=np.random.normal(0, sigma_epsilon, e2_NLA.shape)

    print('Got Shapes')

    epsilon = Table([e1_NLA, e2_NLA, e1_TT, e2_TT, random1, random2], names = ('e1_NLA', 'e2_NLA', 'e1_TT', 'e2_TT', 'e1_noise', 'e2_noise'))
    #epsilon.write('../../../HACC-Y1/GalCat/SRD-Y1/HACC150_3/IA/epsilon_{}_0.6GpAM.asc'.format(i), format='ascii', overwrite=True)    
    epsilon.write('../../../BCM/'+BCM_model+'/GalCat/epsilon_'+str(i)+'_0.6GpAM'+smooth_flag+'_pollux.asc', format='ascii', overwrite=True)    

    print('Done tomo',i)


# In[8]:


ls ../../../BCM/dmb_theta_ej_2


# In[12]:


# Alternative way to do this, using astropy table to read the catalogues instead of numpy. Often runs out of memory for strange reasons.
for i in np.arange(1,6):

    # This sometimes takes huge amount of memory, when astropy table gets bunkers. 
    # Use the next cell if that happens

    # descriptions=['ra_deg', 'dec_deg', 'shear1', 'shear2', 'weight', 'z_true', 'delta', 's11', 's22', 's12']


    #SkySim
    #catalog = Table.read('/global/cscratch1/sd/jharno/IA_Infusion/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(i), format='ascii')  
    #catalog = Table.read('/global/cscratch1/sd/jharno/IA_Infusion/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij_smooth.dat'.format(i), format='ascii')  

    #V0
    #catalog = Table.read('../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij_smooth.dat'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc'.format(i), format='ascii')  

    #V1
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias.asc'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_1Mpc.asc'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.1Mpc.asc'.format(i), format='ascii')  
    #catalog = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta1_smooth0.1Mpc.asc'.format(i), format='ascii')
    catalog = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta1_smooth0.5Mpc.asc'.format(i), format='ascii')


    print('Got Gatalogue')

    s11=catalog['col8']
    s22=catalog['col9']
    s12=catalog['col10']
    z=catalog['col6']

    e1_NLA=Epsilon1_NLA(cosmo,z,Aia,rho_crit,s11,s22)
    e2_NLA=Epsilon2_NLA(cosmo,z,Aia,rho_crit,s12)

    e1_TT=Epsilon1_TT(cosmo,z,C2,rho_crit,s11,s22)
    e2_TT=Epsilon2_TT(cosmo,z,C2,rho_crit,s11,s22,s12)

    random1=np.random.normal(0, sigma_epsilon, e1_NLA.shape)
    random2=np.random.normal(0, sigma_epsilon, e2_NLA.shape)

    print('Got Shapes')

    epsilon = Table([e1_NLA, e2_NLA, e1_TT, e2_TT, random1, random2], names = ('e1_NLA', 'e2_NLA', 'e1_TT', 'e2_TT', 'e1_noise', 'e2_noise'))

    #epsilon.write('/global/cscratch1/sd/jharno/epsilon_{}.dat'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('/global/cscratch1/sd/jharno/epsilon_{}_smooth.dat'.format(i), format='ascii', overwrite=True)    

    #***
    #V0:
    #epsilon.write('../../GalCat/StageIV_nz/V0/epsilon_{}_0.06GpAM_V2.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/StageIV_nz/V0/epsilon_{}_0.06GpAM_smooth_V2.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.5.asc'.format(i), format='ascii', overwrite=True)    

    #***
    #V1
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.6GpAM_linear_bias_smooth.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth_1Mpc.asc'.format(i), format='ascii', overwrite=True)    
    #epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_smooth1_bta1_smooth01Mpc.asc'.format(i), format='ascii', overwrite=True)    
    epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_smooth1_bta1_smooth05Mpc.asc'.format(i), format='ascii', overwrite=True)    

    print('Done tomo',i)


# In[ ]:




