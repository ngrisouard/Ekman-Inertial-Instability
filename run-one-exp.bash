#!/bin/bash
# arguments:
# $1: value of viscosity in m2/s
# $2: value of Ro (including negative sign if any)
# $3: if there is a 3rd argument, we run Dedalus

crit=-1.  # critical Rossby number; decides if EL or EII
if (( $(echo "$2 > $crit" |bc -l) )); then
    echo '>>> Ro='$2', this is an Ekman spiral <<<'
    plotname=EL_plots.py
else
    echo '>>> Ro='$2', this is an Ekman-Inertial Instability <<<'
    plotname=EII_plots.py
fi

expname=EXP_nu${1}_Ro${2}
expname=${expname//-/m}  # replace minus signs with m letter
expname=${expname//.}  # delete dots
mkdir $expname
cd $expname

if [ "$#" -eq 3 ]; then  # new Dedalus run
    rm -r *

    cp ../dedalus_1D.py .
    ed -s "dedalus_1D.py" <<< $'g/xnux/s/xnux/'${1}$'/g\nw\nq'
    ed -s "dedalus_1D.py" <<< $'g/xRox/s/xRox/'${2}$'/g\nw\nq'

    python dedalus_1D.py
fi

cp ../$plotname .
ed -s "$plotname" <<< $'g/xnux/s/xnux/'${1}$'/g\nw\nq'
ed -s "$plotname" <<< $'g/xRox/s/xRox/'${2}$'/g\nw\nq'
python $plotname
