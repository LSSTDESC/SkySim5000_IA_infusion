from astropy.cosmology import FlatLambdaCDM
from string import Template

################################ Lensing parameters #####################################

# Map index
imap = 1
# The index of the source redshifts
source_redshifts = [5, 15, 23, 28]

################################ Cosmology Pars #########################################
Om0    = 0.2648
H0     = 71.0
#ns     = 0.973
#sigma8 = 0.799
#Ob0    = 0.048
#pert_params     =  {'flat': True, 'H0': H0, 'Om0': Om0, 'Ob0': Ob0, 'sigma8': sigma8, 'ns': ns, 'relspecies':False}

################################ Base Directory #########################################
baseDirectory   = "/global/cfs/cdirs/lsst/groups/CS/mass_sheets/"
############################# Mass Maps Directory #######################################
massMapsNames   =  Template(baseDirectory+'/density_maps_*_dens.bin')
############################# Kappa Maps Directory ######################################
kappaDir  = "../../kappa/"
############################# Shear Maps Directory ######################################
shearDir  = "../../shear/"
############################# Galaxy Catalogue Directory ######################################
GalDir  = "IA-infusion/SkySim5000/GalCat/"
############################# Delta Maps Directory ######################################
deltaDir = "IA-infusion/SkySim5000/density/"
############################# Tidal Field Maps Directory ######################################
tidalDir = "IA-infusion/SkySim5000/tidalfield/"