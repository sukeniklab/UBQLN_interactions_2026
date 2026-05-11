# Analysis scripts and output associated with "Interactions between folded domains and disordered regions shape ubiquillin structural topology and function" #

## by Jess Niblo, Nirbhik Acharya, Max Watkins, Thuy P. Dao, Carlos A. Castañeda, Shahar Sukenik ##

Scripts to analyze data and prepare figures from the manuscript available at [https://www.biorxiv.org/content/10.64898/2026.03.13.711692v1](https://www.biorxiv.org/content/10.64898/2026.03.13.711692v1)

Representative open/closed trajectories and reconstructed ensembles are available at [https://zenodo.org/records/19904341](https://zenodo.org/records/19904341)


Repository Structure: 
```
.
├── Main_Figs/
├── SI_Figs/
└── Simulation_Input/
```

### Main_Figs/
Scripts and all processed CSVs or .dat files used to generate all figures within the main text. Each subdirectory is titled following the figure it is attached to, with the scripts named to generate the affiliated subplot. 

### SI_Figs/
Scripts and all processed CSVs or .dat files used to generate all figures within the supplemental text. Each subdirectory is titled following the figure it is attached to, with the scripts named to generate the affiliated subplot. 

### Simulation_Input/
Initial pdbs and forcefield parameters to perform the CALVADOS simulations. Simulations were performed using the CALVADOS3 COM code available at [_2024_Cao_CALVADOSCOM](https://github.com/KULL-Centre/_2024_Cao_CALVADOSCOM/tree/main). Boundaries of folded domains are specified in Supplementary Information Table S5.

