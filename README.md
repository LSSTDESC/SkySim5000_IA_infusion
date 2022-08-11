# SkySim5000_IA_infusion
Repo to store code and documentation for the IA infusion


This repo contains a number of jupyter notebooks meant to validate the various SkySim5000 mock data and the predictions.
The IA-free data has been validated with Namaster (see the Namaster notebook) and with TreeCorr (see the TreeCorr notebook).

The IA infusion is validated in three steps: the Infusion_IA notebook converts the tidal field quantities s11, s22 and s12 into ellipticities, following either the NLA or the TT model. The notebook Infusion_postprocessing then reads these, reweiths the relative ellipticities with three IA parameters (A_IA, b_TA and C2) and constructs IA-infused mocks. The Treecorr notebook also computes 2PCFs from these mocks. Results are compared in the Compare_Mocks_Theory notebook. 
