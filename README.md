# Ekman Inertial Instability

All code, supporting the manuscript "Ekman-Inertial Instability" by Grisouard & Zemskova.

## Pre-requisites
* All code can run on a personal machine and does not require a supercomputer.
* A Python distribution, with common packages such at NumPy, SciPy, Matplotlib...
* the in-line text editor `ed` if you wish to use the `run-one-exp.bash` script.
* Dedalus: see http://dedalus-project.org/. The version we used was ``.

## Execution

All commands to execute the dedalus and plotting scripts are gathered in the `run-one-exp.bash` file. Adapt them to your OS or shell as needed, or run as such.

Example: you want to run a simulation with a vertical viscosity nu=1e-4 m2/s and a Rossby number Ro=-1.1 (the values we used in the article). If executing for the first time, execute

```$ bash run-one-exp.bash 1e-4 -1.1 X```

where `X` can be anything, this flag being just an instruction to execute dedalus. You can change more parameters directly in `dedalus_1D.py`. If you already ran the simulation and only wish to modify a plotting script (`EII_plots.py` or `EL_plots.py`), you will save yourself some time by omitting the third argument `X`:

```$ bash run-one-exp.bash 1e-4 -1.1```

If you have Ro<-1, the script will call `EII_plots.py`. If not, it will call `EL_plots.py`.
If you change either, do your changes in the file at the root of the repo, `run-one-exp` will do the rest.
