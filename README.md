# Nonlinear Solvent Response and Transferability of Effective Born Radii for Model Charged Sites in Biomolecular Solvation

This repository contains input structures, molecular dynamics scripts, and analysis files for the study of ion water clusters and effective Born radii.

## Project overview

The project examines how the effective Born radius depends on ion charge state and ion separation in finite water clusters. Molecular dynamics simulations were performed for single ion systems and two ion systems. The resulting trajectories were analyzed to calculate cumulative electrostatic potentials, charge response profiles, and continuum model estimates.

The main systems include:

- Single ion systems: K+, Cl−, and Mg2+
- Two ion systems: K+ K+, Cl− Cl−, and K+ Cl−
- Two ion separation distances: d = 1 to 8 Å
- Water cluster radius: 20 Å
- Single ion water clusters: 1118 water molecules
- Two ion water clusters: 1117 water molecules

## Repository structure

```text
24_Born/
├── 1_prepare/
├── 2_MD/
└── 3_paper/
```

## 1_prepare

The `1_prepare` folder contains scripts and generated initial PDB files for molecular dynamics simulations.

This folder is used to generate the initial ion water cluster structures.

Example initial PDB files include:

```text
single_K_sel21A_1118w.pdb
single_Cl_sel21A_1118w.pdb
single_Mg_sel21A_1118w.pdb

pair_K_K_d01A_sel21A_1117w.pdb
pair_K_K_d02A_sel21A_1117w.pdb
pair_K_K_d03A_sel21A_1117w.pdb
pair_K_K_d04A_sel21A_1117w.pdb
pair_K_K_d05A_sel21A_1117w.pdb
pair_K_K_d06A_sel21A_1117w.pdb
pair_K_K_d07A_sel21A_1117w.pdb
pair_K_K_d08A_sel21A_1117w.pdb

pair_Cl_Cl_d01A_sel21A_1117w.pdb
pair_Cl_Cl_d02A_sel21A_1117w.pdb
pair_Cl_Cl_d03A_sel21A_1117w.pdb
pair_Cl_Cl_d04A_sel21A_1117w.pdb
pair_Cl_Cl_d05A_sel21A_1117w.pdb
pair_Cl_Cl_d06A_sel21A_1117w.pdb
pair_Cl_Cl_d07A_sel21A_1117w.pdb
pair_Cl_Cl_d08A_sel21A_1117w.pdb

pair_K_Cl_d01A_sel21A_1117w.pdb
pair_K_Cl_d02A_sel21A_1117w.pdb
pair_K_Cl_d03A_sel21A_1117w.pdb
pair_K_Cl_d04A_sel21A_1117w.pdb
pair_K_Cl_d05A_sel21A_1117w.pdb
pair_K_Cl_d06A_sel21A_1117w.pdb
pair_K_Cl_d07A_sel21A_1117w.pdb
pair_K_Cl_d08A_sel21A_1117w.pdb
```

The file name indicates the ion type, ion separation distance, selected water cluster size, and number of retained water molecules.

## 2_MD

The `2_MD` folder contains molecular dynamics scripts.

The simulations use:

- OpenMM
- CHARMM36 force field
- CHARMM modified TIP3P water model
- NoCutoff for nonbonded interactions
- A spherical wall applied to water oxygen atoms
- Charge scaling simulations for single ion systems
- Full charge and zero charge reference simulations for two ion systems

The zero charge reference simulations keep the ion positions and van der Waals parameters unchanged, while setting the ion charges to zero.

## 3_paper

The `3_paper` folder contains files used for post processing, data analysis, figure preparation, and manuscript writing.

The analyses include:

- Single ion cumulative MD potential
- Potential difference relative to the zero charge reference
- Effective Born radius as a function of charge state
- Born model potential estimated using a constant Born radius
- Radial charge response of water
- Two ion cumulative potential after subtracting the zero charge reference
- Comparison between MD results and continuum model estimates

## Notes

Large trajectory files are excluded from the repository because of file size. The scripts and analysis files are provided to document the workflow and reproduce the main calculations when the trajectory data are available.

## Author

Pei-Kun Yang  
Independent Researcher, Taiwan
