# DFT-Assisted Field Emission Simulations of Carbon Nanotube Cathodes

> First-principles simulations of carbon nanotube (CNT) field emitters combining **PySCF**, **FFT-accelerated electrostatics**, and **Fowler–Nordheim emission theory** for the investigation of electron emission under applied electric fields.

<img width="1774" height="887" alt="Computational_GraphicalAbstract" src="https://github.com/user-attachments/assets/92929fd5-be87-4b60-97d1-c7068e125a52" />

---

## Overview

This repository contains the computational framework I developed during my Master's research for studying **field emission from carbon nanotube (CNT) cathodes** using density functional theory (DFT) and classical electron emission models.

The workflow combines:

- Electronic structure calculations using **PySCF**
- Hybrid DFT calculations accelerated through **RIJCOSX**
- FFT-based evaluation of electrostatic potentials
- Real-space electric field reconstruction
- Work-function extraction
- Field enhancement analysis
- Fowler–Nordheim emission current calculations
- Automated post-processing and visualization

The codes were designed for execution on the **Santos Dumont (SDumont)** supercomputer and are intended to provide a reproducible workflow for first-principles field-emission studies.

It is worth mentioning that the basis sets and functionals used for these simulations sit on the heavier side of computational chemistry, therefore, the availability of CPUs might be a limiting factor of reproducing this framework. 
---

# Scientific Background

Carbon nanotubes are among the most promising cold cathode materials due to their high aspect ratio, chemical stability, and exceptional field enhancement properties.

Instead of relying solely on analytical approximations, this project evaluates the complete electrostatic environment directly from first-principles electronic structure calculations.

The methodology reconstructs the total emission potential in a way that:

- **Hartree potential** is obtained through FFT-accelerated Poisson solvers,
- **Nuclear potential** is evaluated from atomic positions,
- **External potential** represents the applied macroscopic electric field.

The local electric field is then computedallowing direct extraction of:

- Local electric field maps
- Field enhancement factors (β)
- Vacuum level profiles
- Surface electrostatic barriers
- Fowler–Nordheim current densities

---

# Computational Methodology

The electronic structure workflow follows a numerically stable strategy specifically designed for extended π-conjugated carbon systems.

Main features include

- Initial LDA preconditioning
- Hybrid DFT refinement
- RIJCOSX acceleration
- Density purification
- Controlled occupation smearing
- Vacuum-level alignment
- FFT-based Hartree potential evaluation
- Real-space electrostatic reconstruction

The resulting electrostatic potential is used to evaluate field emission properties through classical Fowler–Nordheim theory.

---

# Repository Structure

```
.
├── input/
│   ├── xyz structures
│   ├── basis sets
│   ├── simulation parameters
│   └── external electric fields
│
├── pyscf/
│   ├── SCF calculations
│   ├── density generation
│   ├── work-function calculations
│   └── electrostatic analysis
│
├── fft/
│   ├── Hartree potential solver
│   ├── reciprocal-space routines
│   └── grid generation
│
├── field_emission/
│   ├── Fowler–Nordheim calculations
│   ├── β-factor extraction
│   └── emission current estimation
│
├── visualization/
│   ├── plots
│   ├── figures
│   └── publication-quality graphics
│
└── results/
    ├── densities
    ├── potentials
    ├── electric fields
    ├── work functions
    └── emission currents
```

---

# Main Features

- Density Functional Theory calculations using **PySCF**
- Support for large CNT systems
- FFT-accelerated Hartree potential reconstruction
- Real-space electrostatic potential evaluation
- Local electric field mapping
- Automatic work-function extraction
- Field enhancement (β) calculation
- Fowler–Nordheim emission current estimation
- Publication-quality visualization tools
- HPC-ready execution on SDumont

---

# Computational Workflow

```text
CNT Geometry
      │
      ▼
PySCF DFT Calculation
      │
      ▼
Electron Density
      │
      ▼
FFT Hartree Solver
      │
      ▼
Total Electrostatic Potential
      │
      ▼
Electric Field
      │
      ▼
Field Enhancement (β)
      │
      ▼
Work Function
      │
      ▼
Fowler–Nordheim Analysis
      │
      ▼
Emission Current Density
```

---

# Software

Main packages used throughout the project include

- Python 3
- PySCF
- NumPy
- SciPy
- Matplotlib
- h5py

Calculations were executed on the **Santos Dumont (SDumont)** HPC cluster.

---

# Applications

The framework can be adapted to investigate

- Carbon nanotubes
- Functionalized CNT emitters
- Metal nanoparticle-decorated CNTs
- Vacuum nanoelectronics
- Cold cathodes
- Electron sources
- Space propulsion emitters
- Nanoelectronic devices

---

# Research Goals

The purpose of this project is to establish a fully first-principles workflow capable of connecting

- electronic structure,
- electrostatics,
- local field enhancement,
- work-function modification,

to experimentally relevant field-emission observables.

---

# Future Developments

Current and planned extensions include

- plasma generation from emitted electrons
- Monte Carlo electron transport
- Townsend avalanche simulations
- coupling with BOLSIG+ swarm data
- Langmuir probe modelling
- space-environment plasma interactions
- multiscale DFT → plasma simulations

---

# Citation

If you use this repository or build upon this work, please cite the associated Master's dissertation:

Santos, Ana Carolina Dias de Lima
ENHANCEMENT OF THE ELECTRICAL PROPERTIES OF CARBON NANOTUBES BY THE INCORPORATION OF GOLD NANOPARTICLES: DEVELOPMENT OF A FIELD EMISSION DEVICE AS
A CATHODE ON ION GRIDDED THRUSTERS/ Ana Carolina Dias de Lima Santos. – Rio de Janeiro: UFRJ/COPPE, 2026.

# License

This repository is released under the MIT License unless otherwise specified.
