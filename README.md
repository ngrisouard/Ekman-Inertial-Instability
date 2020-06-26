# Ekman Inertial Instability

All code, supporting the manuscript "Ekman-Inertial Instability" by Grisouard & Zemskova.

## Pre-requisites
* All code can run on a personal machine and does not require a supercomputer.
* A Python distribution, with common packages such at NumPy, SciPy, Matplotlib...
* the in-line text editor `ed` if you wish to use the `run-one-exp.bash` script.
* Dedalus: see http://dedalus-project.org/. The version we used was ``.

## Execution

All commands to execute the dedalus and plotting scripts are gathered in the `run-one-exp.bash` file. Adapt them to your OS or shell as needed, or run as such.

Example: you want to run a simulation with a vertical viscosity $\nu=2\times 10^{-3}$ m$^2$.s$^{-1}$ and a Rossby number $Ro=-1.2$. I executing for the first time, execute

`$ bash run-one-exp.bash 2e-3 -1.2 X`

where `X` can be anything, this flag being just an instruction to execute dedalus. If you already ran the simulation and only wish to modify a plotting script (`EII_plots.py` or `EL_plots.py`), you will save yourself some time by omitting the third argument `X`.
