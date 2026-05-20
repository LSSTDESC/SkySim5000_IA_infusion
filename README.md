# SkySim5000_IA_infusion
Repo to store code and documentation for the IA infusion method

This repo contains a number of jupyter notebooks meant to validate the various mock data and the predictions. Here is the overall production & validation pipeline, starting with a series of mass sheets from the OuterRim simulation, filling the light-cone up to z = 3.

1-The notebook MiraTitanBorn.ipynb ray-traces the SkySim5000 mass sheets and produces shear and kappa maps. 

2-The notebook SkySim5000_IA.ipynb computes projected tidal field maps s11, s22 and s12 from the mass sheets. 

3-The notebook PopulateGal.ipynb generates galaxy lensing catalogues given an n(z), n_gal, and an tracer model (random position or linearly tracing the projected matter distribution)

4a-The notebook AddDeltaSij.ipynb adds the delta, s11, s22 and s12 values to the lensing catalogue.
4b-Catalogues are concatenated from indivual mass planes to full tomographic samples.

5-The Infusion_IA.ipynb notebook converts the tidal field quantities s11, s22 and s12 into intrinsic ellipticities, following the NLA, delta-NLA, TT, delta_TT model or any combination of these. 

6-The notebook Infusion_postprocessing.ipynb then reads these, reweiths the relative ellipticities with three IA parameters (A_IA, b_TA and C2) and constructs IA-infused mocks. 

7-The Treecorr.ipynb notebook computes 2PCFs from these mocks (the GG-noIA, GG, GI and II terms). 

8-Predictions are computed with CCL in theo_predictions.ipynb

9-Results are compared in Compare_Mocks_Theory.ipynb 


The IA-free data has also been validated with Namaster (see the Namaster notebook)