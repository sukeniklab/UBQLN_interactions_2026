This code requires the [IMP package from the Sali lab](https://integrativemodeling.org/)

To run this pipeline: 

1. Run run_foxs_trial.py on your trajectory. This will produce a directory (not included in this upload due to size constraints) where each frame is saved as a seperate .pdb and .dat file. 

2. Run reconstruct_ensemble.py. This will perform a linear fit on open/closed simulations to experimental data and produce a reconstructed ensemble in .pdb format. 