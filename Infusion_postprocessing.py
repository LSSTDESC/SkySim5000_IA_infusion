#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from astropy.table import Table


# We use the constants for different IA models as inputs

# In[2]:


A_IA = 1.0
b_TA = 0.0
C_2  = 0.0


# The intensity of the NLA, delta-NLA and TT models can be chosen by tweaking the numbers above, this can probably be changed for some calculation involving weights given to each model but that's up for discussion, for now, we setup for the observed epsilons with and without noise, using the provided parameters

# In[5]:


for i in range(1,6):
#for i in np.arange(5)+1:

    #V0
    #catalog = Table.read('../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij.dat'.format(i),
    #catalog = Table.read('../../GalCat/StageIV_nz/V0/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij_smooth.dat'.format(i),
    catalog = Table.read('../../GalCat/SRD-Y1/random_pos/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc'.format(i),

    #V1
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias.asc'.format(i),
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i),
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_1Mpc.asc'.format(i),
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i),
    #catalog = Table.read('/global/homes/j/jharno/IA-infusion/SkySim5000/GalCat/StageIV_nz/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_den_Sij_smooth.dat'.format(i),
                     format='ascii')  # descriptions=['ra_deg', 'dec_deg', 'shear1', 'shear2', 'weight', 'z_true', 'delta', 's11', 's22', 's12'])

    #V0
    #epsilons = Table.read('../../GalCat/StageIV_nz/V0/epsilon_{}_0.06GpAM.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V0/epsilon_{}_0.06GpAM_V2.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V0/epsilon_{}_0.06GpAM_smooth_V2.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    epsilons = Table.read('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc_nofactor2.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']

    #V1
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.6GpAM_linear_bias_smooth.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth_1Mpc.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']


    ra=np.asarray(catalog['col1'])
    dec=np.asarray(catalog['col2'])
    weight=np.asarray(catalog['col5'])
    z=np.asarray(catalog['col6'])
    gamma1 = np.asarray(catalog['col3'])
    gamma2 = np.asarray(catalog['col4'])
    e1_NLA = np.asarray(epsilons['e1_NLA'])
    e2_NLA = np.asarray(epsilons['e2_NLA'])
    e1_TT = np.asarray(epsilons['e1_TT'])
    e2_TT = np.asarray(epsilons['e2_TT'])
    e1_noise = np.asarray(epsilons['e1_noise'])
    e2_noise = np.asarray(epsilons['e2_noise'])
    delta = np.asarray(catalog['col7'])
    epsilon_NLA = np.empty(e1_NLA.shape, dtype = complex)
    epsilon_NLA.real = A_IA * e1_NLA  
    epsilon_NLA.imag = A_IA * e2_NLA  
    epsilon_TT = np.empty(e1_TT.shape, dtype = complex)
    epsilon_TT.real = C_2 * e1_TT  
    epsilon_TT.imag = C_2 * e2_TT
    epsilon_random = np.empty(e1_TT.shape, dtype = complex)
    epsilon_random.real = e1_noise
    epsilon_random.imag = e2_noise
    g = np.empty(gamma1.shape, dtype=complex)
    g.real = gamma1
    g.imag = gamma2


    #calibrate all IA:
    # Needed after finding the factor of 2.0 bug in the e_IA2
    epsilon_NLA*=1.5
    epsilon_TT*=1.5

    #calibrate delta:
    #delta /=2.5
    #delta*=0.0

    #calibrate the low-redshift e_TT:
    epsilon_TT[z<0.5]/=2.5

    #V1 mocks further require an overall calibration (not sure why)
    #epsilon_TT /=20.0


    #And finally, we calculate the observed epsilons using the summation as in equation 11 of the paper
    # The V1 (linear bias) mocks already have the delta-weighted term with b_ta=1.0, so we keep it or we undo it. 
    # In the later case, we must use the same b_TA parameter as that used in the construction of the mocks

    # 1-Keep delta-weights:
    epsilon_IA_TATT = epsilon_NLA + epsilon_TT
    # 2-Undo delta-weights:
    #epsilon_IA_TATT = epsilon_NLA / (1 + b_TA * delta) + epsilon_TT
    #epsilon_IA_TATT /= (1 + b_TA * delta)
    # 3-Change from b-TA = 1 to b_TA = 0 or 2.0 
    #epsilon_IA_TATT *= (1 + b_TA * delta)/(1 + 1.0 * delta)

    epsilon_int = (epsilon_IA_TATT + epsilon_random)/(1 + np.conjugate(epsilon_IA_TATT) * epsilon_random)
    epsilon_obs = (epsilon_int + g)/(1 + epsilon_int * np.conjugate(g))
    epsilon_obs_no_noise = (epsilon_IA_TATT + g)/(1 + epsilon_IA_TATT * np.conjugate(g))

    fin_epsilon = Table([ra,dec,z,gamma1, gamma2, weight,delta,epsilon_obs.real,epsilon_obs.imag,epsilon_obs_no_noise.real,epsilon_obs_no_noise.imag,epsilon_IA_TATT.real,epsilon_IA_TATT.imag], names = ('ra','dec','z','g1', 'g2', 'weight','delta','e_obs_1','e_obs_2', 'e_obs_no_noise_1','e_obs_no_noise_2','e_IA_TATT_1', 'e_IA_TATT_2' ))
    #fin_epsilon.write('/global/cscratch1/sd/jharno/fin_epsilon_IA_{}.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('/global/cscratch1/sd/jharno/fin_epsilon_IA_{}_linear_bias.dat'.format(i), format='ascii', overwrite=True)    

    #V0:
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth0.1Mpc.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth0.5Mpc_nofactor2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/NLA/galcat_skysim5000_noIA_tomo{}.dat'.format(i), format='ascii', overwrite=True)    
    fin_epsilon.write('../../GalCat/SRD-Y1/NLA/galcat_skysim5000_IAIp1_tomo{}_nofactor2.dat'.format(i), format='ascii', overwrite=True)    

    #V0-TT                     
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_V2corr.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_smooth_V2corr.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM_TT_smooth_V3corr.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_TT_smooth0.1Mpc.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_TT_smooth0.5Mpc_m1.dat'.format(i), format='ascii', overwrite=True)    

    #V1:
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(i), format='ascii', overwrite=True)    

    #V1-NLA:
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_NLA.dat'.format(i), format='ascii', overwrite=True)    

    print('Done tomo',i)


# In[33]:


for i in range(1,6):
    ra,dec,z,gamma1,gamma2 = np.loadtxt('../../GalCat/HOD/FromCharlie/GalCat_tomo'+str(i)+'_06GpAM_new.dat',usecols = (3,6,1,8,9), unpack=True)
    epsilons = Table.read('../../GalCat/HOD/FromCharlie/epsilon_{}_0.6GpAM.asc'.format(i), format='ascii')

    #ra=np.asarray(catalog[:,4])
    #dec=np.asarray(catalog[:,7])
    weight=np.asarray(np.ones(np.shape(ra)[0]))
    #z=np.asarray(catalog[:,2])
    #gamma1 = np.asarray(catalog[:,8])
    #gamma2 = np.asarray(catalog[:,9])
    e1_NLA = np.asarray(epsilons['e1_NLA'])
    e2_NLA = np.asarray(epsilons['e2_NLA'])
    e1_TT = np.asarray(epsilons['e1_TT'])
    e2_TT = np.asarray(epsilons['e2_TT'])
    e1_noise = np.asarray(epsilons['e1_noise'])
    e2_noise = np.asarray(epsilons['e2_noise'])
    #delta = np.asarray(catalog[:,6])
    delta = np.asarray(np.ones(np.shape(ra)[0]))
    epsilon_NLA = np.empty(e1_NLA.shape, dtype = complex)
    epsilon_NLA.real = A_IA * e1_NLA *(-1.0)
    epsilon_NLA.imag = A_IA * e2_NLA *(-1.0)
    epsilon_TT = np.empty(e1_TT.shape, dtype = complex)
    epsilon_TT.real = C_2 * e1_TT *(-1.0)
    epsilon_TT.imag = C_2 * e2_TT *(-1.0)
    epsilon_random = np.empty(e1_TT.shape, dtype = complex)
    epsilon_random.real = e1_noise
    epsilon_random.imag = e2_noise
    g = np.empty(gamma1.shape, dtype=complex)
    g.real = gamma1
    g.imag = gamma2


    #calibrate all IA:
    # Needed after finding the factor of 2.0 bug in the e_IA2
    epsilon_NLA*=1.5
    epsilon_TT*=1.5


    #------------------
    #calibrate delta:
    #delta*=0.0

    #calibrate HOD_NLA:
    epsilon_NLA/=np.sqrt(2.0) 

    #calibrate TT:
    # for TT: 
    #epsilon_TT[z<0.5]/=2.5 


    #calibrate HOD_TT:
    # for TT: 
    #epsilon_TT[z<0.75]/=4.0 
    #epsilon_TT/=7

    # for and delta-TT
    #epsilon_TT[z<0.5]/=5.0 
    #epsilon_TT/=7 
    #------------------

    #And finally, we calculate the observed epsilons using the summation as in equation 11 of the paper
    # The V1 mocks already sample the galaxy by linearly tracing the matter field, so they already have the 
    # delta-weighted term. 
    # 1-Keep as is:
    epsilon_IA_TATT = epsilon_NLA + epsilon_TT
    # 2-Undo delta-weight:
    #epsilon_IA_TATT = epsilon_NLA / (1 + b_TA * delta) + epsilon_TT
    #epsilon_IA_TATT /= (1 + b_TA * delta) 
    # 3-Change from b_TA = 1 to b_TA = 0 or 2.0 
    #epsilon_IA_TATT *= (1 + b_TA * delta)/(1 + 1.0 * np.where(delta >=  -0.999, delta, -0.999))



    epsilon_int = (epsilon_IA_TATT + epsilon_random)/(1 + np.conjugate(epsilon_IA_TATT) * epsilon_random)
    epsilon_obs = (epsilon_int + g)/(1 + epsilon_int * np.conjugate(g))
    epsilon_obs_no_noise = (epsilon_IA_TATT + g)/(1 + epsilon_IA_TATT * np.conjugate(g))

    fin_epsilon = Table([ra,dec,z,gamma1, gamma2, weight,delta,epsilon_obs.real,epsilon_obs.imag,epsilon_obs_no_noise.real,epsilon_obs_no_noise.imag,epsilon_IA_TATT.real,epsilon_IA_TATT.imag], names = ('#ra','dec','z','g1', 'g2', 'weight','delta','e_obs_1','e_obs_2', 'e_obs_no_noise_1','e_obs_no_noise_2','e_IA_TATT_1', 'e_IA_TATT_2' ))    
    #fin_epsilon.write('../../GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_V2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_nofactor2.dat'.format(i), format='ascii', overwrite=True)    
    fin_epsilon.write('../../GalCat/HOD/FromCharlie/fin_epsilon_TT_{}_0.6GpAM_nofactor2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/HOD/FromCharlie/fin_epsilon_IA_{}_0.6GpAM_TT.dat'.format(i), format='ascii', overwrite=True)    


    print('Done tomo',i)


# In[6]:


#IA tidal field:
#smooth_flag='' #no smoothing
#smooth_flag='_0.5Mpc'

for i in range(1,6):
#for i in np.arange(5)+1:

    #SkySim5000
    #V0
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/random_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc.asc')

    #V1
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias.asc'.format(i),
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc'.format(i),
    #catalog = Table.read('../../GalCat/StageIV_nz/V1/GalCat_tomo{}_All_0.06GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_1Mpc.asc'.format(i),
    #catalog = np.loadtxt('../../GalCat/StageIV_nz/V1/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth.asc')
    #catalog = np.loadtxt('../../GalCat/StageIV_nz/V1/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth_bta2.asc')
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta1_smooth0.5Mpc.asc')
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_bta2_smooth0.1Mpc.asc')
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta1_smooth0.5Mpc.asc')
    #catalog = np.loadtxt('../../GalCat/SRD-Y1/lin_bias_pos/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_linear_bias_smooth1_bta2_smooth0.5Mpc.asc')

    #V0
    #epsilons = Table.read('../../GalCat/StageIV_nz/V0/epsilon_{}_0.06GpAM.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/SRD-Y1/random_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/SRD-Y1/random_pos/epsilon_'+str(i)+'_0.6GpAM_smooth0.5Mpc.asc', format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/SRD-Y1/random_pos/epsilon_'+str(i)+'_0.6GpAM_smooth0.5Mpc_nofactor2.asc', format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']

    #V1
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.6GpAM_linear_bias_smooth.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.6GpAM_linear_bias_smooth_bta2.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = np.loadtxt('../../GalCat/StageIV_nz/V1/epsilon_'+str(i)+'_0.6GpAM_linear_bias_smooth.asc')  #descriptions=['e1','e2','e1_noise','e2_noise']
    #epsilons = Table.read('../../GalCat/StageIV_nz/V1/epsilon_{}_0.06GpAM_linear_bias_smooth_1Mpc.asc'.format(i), format='ascii')  #descriptions=['e1','e2','e1_noise','e2_noise']    
    #epsilons = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_smooth0.5Mpc.asc'.format(i), format='ascii')
    #epsilons = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_smooth_bta2_0.1Mpc.asc'.format(i), format='ascii')
    #epsilons = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_smooth1_bta1_smooth05Mpc_nofactor2.asc'.format(i), format='ascii')
    #epsilons = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_smooth1_bta2_smooth05Mpc.asc'.format(i), format='ascii')

    #BCM:
    BCM_model = 'dmo'
    catalog = np.loadtxt('../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth'+smooth_flag+'.asc')
    epsilons = Table.read('../../../BCM/'+BCM_model+'/GalCat/epsilon_{}_0.6GpAM_smooth'+smooth_flag+'.asc'.format(i), format='ascii')
    #catalog = np.loadtxt('../../../BCM/'+BCM_model+'/GalCat/GalCat_tomo'+str(i)+'_All_0.6GpAM_RA_Dec_g1_g2_w_z_den_Sij_smooth0.5Mpc_linear_bias.asc')
    #epsilons = Table.read('../../../BCM/'+BCM_model+'/GalCat/epsilon_{}_0.6GpAM_smooth0.5Mpc_linear_bias.asc'.format(i), format='ascii')


    ra=np.asarray(catalog[:,0])
    dec=np.asarray(catalog[:,1])
    weight=np.asarray(catalog[:,4])
    z=np.asarray(catalog[:,5])
    gamma1 = np.asarray(catalog[:,2])
    gamma2 = np.asarray(catalog[:,3])
    e1_NLA = np.asarray(epsilons['e1_NLA'])
    e2_NLA = np.asarray(epsilons['e2_NLA'])
    e1_TT = np.asarray(epsilons['e1_TT'])
    e2_TT = np.asarray(epsilons['e2_TT'])
    e1_noise = np.asarray(epsilons['e1_noise'])
    e2_noise = np.asarray(epsilons['e2_noise'])
    delta = np.asarray(catalog[:,6])
    epsilon_NLA = np.empty(e1_NLA.shape, dtype = complex)
    epsilon_NLA.real = A_IA * e1_NLA  
    epsilon_NLA.imag = A_IA * e2_NLA  
    epsilon_TT = np.empty(e1_TT.shape, dtype = complex)
    epsilon_TT.real = C_2 * e1_TT  
    epsilon_TT.imag = C_2 * e2_TT
    epsilon_random = np.empty(e1_TT.shape, dtype = complex)
    epsilon_random.real = e1_noise
    epsilon_random.imag = e2_noise
    g = np.empty(gamma1.shape, dtype=complex)
    g.real = gamma1
    g.imag = gamma2


    #calibrate all IA:
    epsilon_NLA*=1/2
    epsilon_TT*=1/2


    #------------------
    #calibrate delta:
    #delta*=0.0

    #calibrate TT:
    # for TT: 
    epsilon_TT[z<0.5]/=2.5 

    # calibrate delta-TT
    #epsilon_TT[z<0.5]/=5.0 
    #epsilon_TT/=7 
    #------------------

    #And finally, we calculate the observed epsilons using the summation as in equation 11 of the paper
    # The V1 mocks already sample the galaxy by linearly tracing the matter field, so they already have the 
    # delta-weighted term. 
    # 1-Keep as is:
    epsilon_IA_TATT = epsilon_NLA + epsilon_TT
    # 2-Undo delta-weight:
    #epsilon_IA_TATT = epsilon_NLA / (1 + b_TA * delta) + epsilon_TT
    #epsilon_IA_TATT /= (1 + b_TA * delta) 
    # 3-Change from b_TA = 1 to b_TA = 0 or 2.0, uncomment the following: 
    #epsilon_IA_TATT *= (1 + b_TA * delta)/(1 + 1.0 * np.where(delta >=  -0.999, delta, -0.999))



    epsilon_int = (epsilon_IA_TATT + epsilon_random)/(1 + np.conjugate(epsilon_IA_TATT) * epsilon_random)
    epsilon_obs = (epsilon_int + g)/(1 + epsilon_int * np.conjugate(g))
    epsilon_obs_no_noise = (epsilon_IA_TATT + g)/(1 + epsilon_IA_TATT * np.conjugate(g))

    fin_epsilon = Table([ra,dec,z,gamma1, gamma2, weight,delta,epsilon_obs.real,epsilon_obs.imag,epsilon_obs_no_noise.real,epsilon_obs_no_noise.imag,epsilon_IA_TATT.real,epsilon_IA_TATT.imag], names = ('ra','dec','z','g1', 'g2', 'weight','delta','e_obs_1','e_obs_2', 'e_obs_no_noise_1','e_obs_no_noise_2','e_IA_TATT_1', 'e_IA_TATT_2' ))

    #fin_epsilon.write('/global/cscratch1/sd/jharno/fin_epsilon_IA_{}.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('/global/cscratch1/sd/jharno/fin_epsilon_IA_{}_linear_bias.dat'.format(i), format='ascii', overwrite=True)    

    #V0:
    #fin_epsilon.write('../../GalCat/StageIV_nz/V0/fin_epsilon_IA_{}_0.06GpAM.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth0.1Mpc.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth0.1Mpc_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth0.5Mpc_TT.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/random_pos/fin_epsilon_IA_{}_0.6GpAM_smooth0.5Mpc_TT_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/NLA/galcat_skysim5000_noIA_tomo{}.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/NLA/galcat_skysim5000_IA_tomo{}_nofactor2.dat'.format(i), format='ascii', overwrite=True)    


    #V1-deltaNLA:
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.06GpAM_linear_bias_smooth_1Mpc.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.5Mpc.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.5Mpc_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.5Mpc_bta2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.1Mpc_bta2_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.5Mpc_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.5Mpc_TT.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.1Mpc_TT_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/deltaNLA/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_noIA_bta1_tomo{}.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/deltaNLA/galcat_skysim5000_AIAp1_bta1_tomo{}_nofactor2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta1_m1.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta1_111.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta2_TT.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta1_TT_V3.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/deltaTT/galcat_skysim5000_C2p1_bta1_tomo{}_nofactor2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/deltaTT/galcat_skysim5000_C2p1_bta1_tomo{}_V2_nofactor2.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta1_TT_V3_m1.dat'.format(i), format='ascii', overwrite=True)    


    #Change bta:
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth0.5Mpc_bta1_to_bta0.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/SRD-Y1/lin_bias_pos/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth1_smooth0.5Mpc_bta1_to_bta0.dat'.format(i), format='ascii', overwrite=True)    

    #V1-TT:
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_V3.dat'.format(i), format='ascii', overwrite=True)    
    #fin_epsilon.write('../../GalCat/StageIV_nz/V1/fin_epsilon_IA_{}_0.6GpAM_linear_bias_smooth_TT_nodelta.dat'.format(i), format='ascii', overwrite=True)    

    #BCM:
    fin_epsilon.write('../../../BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth'+smooth_flag+'.dat'.format(i), format='ascii', overwrite=True)
    #fin_epsilon.write('../../../BCM/'+BCM_model+'/GalCat/fin_epsilon_NLA_{}_0.6GpAM_smooth0.5Mpc_linear_bias.dat'.format(i), format='ascii', overwrite=True)

    print('Done tomo',i)


# In[9]:


# HACC catalogues
from astropy.io import fits

nside=8192
n_shells=51
seed=0

#BCM_model='dmb'
#BCM_model='dmb_dmo'
#BCM_model='dmb_Mc4e14'
BCM_model='dmb_theta_ej6'

# dndz_flag='Euclid_dndz_fu08_bin1-5'
dndz_flag='SRDv1_nz'
kappa_dir=dndz_flag
folder_path='/pscratch/sd/j/jmena/HOS_Sims_baryonified/'+BCM_model+'/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
#folder_path='/pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
filename='kappa_hacc_seed'+str(seed)+'_nside'+str(nside)+'_'
nshells=51
#BCM_model='dmb'
#BCM_model='dmo'
BCM_model='dmb_theta_ej_6'

#IA tidal field:
#smooth_flag='' #no smoothing

for i in range(1,6):
    smooth_flag='_0.5Mpc'
    #smooth_flag=''
    #filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1_imap'+str(i-1)+'.fits'          # catalogues with tidal field not smoothed
    filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1'+smooth_flag+'_imap'+str(i-1)+'.fits'   # catalogues with tidal field smoothed
    hdul1=fits.open(filename1)
    data1=hdul1[1].data
    header1=hdul1[1].header
    #smooth_flag='_smooth0.5Mpc'
    #smooth_flag=''


    data1_dict={}
    for column in data1.columns:
        #if column.name in ['ra','dec', 'z', 'gamma1', 'gamma2']:
        if column.name in ['ra','dec', 'z', 'gamma1', 'gamma2','delta']:
            data1_dict[column.name]=data1[column.name]

    print('got cats, downsampling...')

    # Let's downsample to 0.6 gal/arcmin**2
    gal_den=0.6;
    import random

    selection1=sorted(random.sample(range(len(data1_dict['ra'])),int(gal_den*(4*np.pi*(180/np.pi)**2/8*60**2))))

    data1_dict_downsampled={}
    for column in data1_dict.keys():
        data1_dict_downsampled[column]=data1_dict[column][selection1]

    print(len(data1_dict_downsampled['ra'])/(4*np.pi*(180/np.pi)**2/8*60**2)) # this should print 0.6

    #epsilons = Table.read('../../../HACC-Y1/GalCat/SRD-Y1/HACC150_3/IA/epsilon_{}_0.6GpAM.asc'.format(i), format='ascii')
    #epsilons = np.loadtxt('../../../HACC-Y1/GalCat/SRD-Y1/HACC150_3/IA/epsilon_{}_0.6GpAM.asc'.format(i), skiprows=1)
    epsilons = np.loadtxt('../../../BCM/'+BCM_model+'/GalCat/epsilon_'+str(i)+'_0.6GpAM'+smooth_flag+'_pollux.asc', skiprows=1)    
    #epsilons = Table.read('../../../BCM/'+BCM_model+'/GalCat/epsilon_{}_0.6GpAM_smooth0.5Mpc_pollux.asc'.format(i), format='ascii')    
    #epsilons = Table.read('../../GalCat/SRD-Y1/lin_bias_pos/epsilon_{}_0.6GpAM_linear_bias_smooth1_bta2_smooth05Mpc.asc'.format(i), format='ascii')

    print('got shapes:', np.shape(epsilons))

    ra=np.asarray(data1_dict_downsampled['ra'])
    dec=np.asarray(data1_dict_downsampled['dec'])
    weight=np.asarray(np.ones(ra.shape))
    z=np.asarray(data1_dict_downsampled['z'])
    gamma1 = np.asarray(data1_dict_downsampled['gamma1'])
    gamma2 = np.asarray(data1_dict_downsampled['gamma2'])

    # e1_NLA = np.asarray(epsilons['e1_NLA'])
    # e2_NLA = np.asarray(epsilons['e2_NLA'])
    # e1_TT = np.asarray(epsilons['e1_TT'])
    # e2_TT = np.asarray(epsilons['e2_TT'])
    # e1_noise = np.asarray(epsilons['e1_noise'])
    # e2_noise = np.asarray(epsilons['e2_noise'])
    e1_NLA = np.asarray(epsilons[selection1,0])
    e2_NLA = np.asarray(epsilons[selection1,1])
    e1_TT = np.asarray(epsilons[selection1,2])
    e2_TT = np.asarray(epsilons[selection1,3])
    e1_noise = np.asarray(epsilons[selection1,4])
    e2_noise = np.asarray(epsilons[selection1,5])


    print('extracted all quantities ')


    delta = np.asarray(np.ones(ra.shape))
    epsilon_NLA = np.empty(e1_NLA.shape, dtype = complex)
    epsilon_NLA.real = A_IA * e1_NLA  
    epsilon_NLA.imag = A_IA * e2_NLA  
    epsilon_TT = np.empty(e1_TT.shape, dtype = complex)
    epsilon_TT.real = C_2 * e1_TT  
    epsilon_TT.imag = C_2 * e2_TT
    epsilon_random = np.empty(e1_TT.shape, dtype = complex)
    epsilon_random.real = e1_noise
    epsilon_random.imag = e2_noise
    g = np.empty(gamma1.shape, dtype=complex)
    g.real = gamma1
    g.imag = gamma2


    #calibrate all IA:
    # Needed for empirical calibration
    epsilon_NLA*=-3/4
    epsilon_TT*=-3/4
    #epsilon_NLA*=-1.0
    #epsilon_TT*=-1.0


    #------------------
    #calibrate delta:
    #delta*=0.0

    #calibrate TT:
    # for TT: 
    #epsilon_TT[z<0.5]/=2.5 

    # calibrate delta-TT
    epsilon_TT[z<0.5]/=5.0 
    epsilon_TT/=7 
    #------------------

    #And finally, we calculate the observed epsilons using the summation as in equation 11 of the paper
    # The V1 mocks already sample the galaxy by linearly tracing the matter field, so they already have the 
    # delta-weighted term. 
    # 1-Keep as is:
    epsilon_IA_TATT = epsilon_NLA + epsilon_TT
    # 2-Undo delta-weight:
    #epsilon_IA_TATT = epsilon_NLA / (1 + b_TA * delta) + epsilon_TT
    #epsilon_IA_TATT /= (1 + b_TA * delta) 
    # 3-Change from b_TA = 1 to b_TA = 0 or 2.0, uncomment the following: 
    epsilon_IA_TATT *= (1 + b_TA * delta)/(1 + 1.0 * np.where(delta <=  -0.999, delta, -0.999))
    #epsilon_IA_TATT *= (1 + b_TA * delta)/(1 + 1.0 * np.where(delta >=  -0.999, delta, -0.999))



    epsilon_int = (epsilon_IA_TATT + epsilon_random)/(1 + np.conjugate(epsilon_IA_TATT) * epsilon_random)
    epsilon_obs = (epsilon_int + g)/(1 + epsilon_int * np.conjugate(g))
    epsilon_obs_no_noise = (epsilon_IA_TATT + g)/(1 + epsilon_IA_TATT * np.conjugate(g))

    fin_epsilon = Table([ra,dec,z,gamma1, gamma2, weight,delta,epsilon_obs.real,epsilon_obs.imag,epsilon_obs_no_noise.real,epsilon_obs_no_noise.imag,epsilon_IA_TATT.real,epsilon_IA_TATT.imag], names = ('ra','dec','z','g1', 'g2', 'weight','delta','e_obs_1','e_obs_2', 'e_obs_no_noise_1','e_obs_no_noise_2','e_IA_TATT_1', 'e_IA_TATT_2' ))
    #fin_epsilon.write('../../../HACC-Y1/GalCat/SRD-Y1/HACC150_3/NLA/galcat_skysim5000_AIAp1_poisson_tomo{}.dat'.format(i), format='ascii', overwrite=True)        
    fin_epsilon.write('../../../BCM/'+BCM_model+'/GalCat/galcat_pollux_AIAp1_poisson_tomo'+str(i)+smooth_flag+'.dat', format='ascii', overwrite=True)        

    print('Done tomo',i)


# In[10]:


from astropy.io import fits
nside=8192
n_shells=101
seed=0
# dndz_flag='Euclid_dndz_fu08_bin1-5'
dndz_flag='SRDv1_nz'
kappa_dir=dndz_flag
folder_path='/pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
filename='kappa_hacc_seed'+str(seed)+'_nside'+str(nside)+'_'
nshells=51
for i in range(1,2):
    filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1_imap'+str(i-1)+'.fits'
    hdul1=fits.open(filename1)
    data1=hdul1[1].data
    header1=hdul1[1].header


# In[7]:


get_ipython().system('head -n 1 ../../../BCM/dmb_theta_ej_3/GalCat/galcat*')


# In[15]:


np.shape(epsilons)


# In[16]:


len(data1_dict['ra'])


# In[6]:


# HACC catalogues
from astropy.io import fits
nside=8192
n_shells=51
seed=0
# dndz_flag='Euclid_dndz_fu08_bin1-5'
dndz_flag='SRDv1_nz'
kappa_dir=dndz_flag
folder_path='/pscratch/sd/j/jmena/HOS_Sims/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir
filename='kappa_hacc_seed'+str(seed)+'_nside'+str(nside)+'_'
nshells=51
BCM_model='dmo'


# In[17]:


BCM_model='dmb_theta_ej6'
n_shells=51
folder_path='/pscratch/sd/j/jmena/HOS_Sims_baryonified/'+BCM_model+'/HACC150_3/shells_z'+str(n_shells)+'_subsampledensitymax1.5_nside'+str(nside)+'/'+kappa_dir

filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1'+smooth_flag+'_imap'+str(i-1)+'.fits'   # catalogues with tidal field smoothed
#filename1 = folder_path+'/'+filename+'catpoisson_interppixels_b1'+smooth_flag+'_imap1.fits'
hdul1=fits.open(filename1)
data1=hdul1[1].data
header1=hdul1[1].header


# In[20]:


header1


# In[8]:


get_ipython().system('ls -l /pscratch/sd/j/jmena/HOS_Sims_baryonified/*/HACC150_3/shells_z51_subsampledensitymax1.5_nside8192/SRDv1_nz/kappa_hacc_seed0_nside8192_catpoisson_interppixels_b1_0.5Mpc_imap0.fits')




# In[ ]:




