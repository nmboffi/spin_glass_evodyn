# Spin Glass Evolutionary Dynamics
This repository contains code to reproduce the results of the publication ``Boffi Nicholas M., Guo Yipei, Rycroft Chris H., Amir Ariel (2023) How microscopic epistasis and clonal interference shape the fitness trajectory in a spin glass model of microbial long-term evolution eLife 12:RP87895 https://doi.org/10.7554/eLife.87895.1``. 

It is also an extensive simulation environment for microbial evolutionary dynamics at microscopic granularity, and may be useful to the grander evolutionary dynamics community as a result.

The folder ``py`` contains many useful routines for processing the output simulation data. Some of these are directly used for producing the publication figures, while others are useful for exploratory data analysis.

## Installation
The main simulation package requires ``openmp`` and ``GSL``; both can be installed using standard package managers.

On Mac, ``openmp`` is not supported by the default C++ compiled provided by XCode. The simplest way to proceed is to install ``gcc`` via a package manager such as ``homebrew`` and compile with the corresponding 

The data analysis routines are written in Python and are built upon standard scienific Python libraries (``numpy``, ``scipy``, ``matplotlib``, etc.). ``numba`` is also used to accelerate some computations; it can be installed using ``pip`` or ``conda``.

## Usage
The simulation package can be compiled by running ``make``.

The executable ``lenski_main`` can be use to run a microbial evolution experiment **in-silico**. By default it splits all replicates over all available threads. If you want to use less, cap the number of threads using the ``OMP_NUM_THREADS`` environment variable.

The executables ``lenski_vary_epi`` and ``lenski_vary_clonal`` may be used to reproduce the results of the eLife publication, sweeping over the strength of epistasis and the strength of clonal interference.

## Referencing
