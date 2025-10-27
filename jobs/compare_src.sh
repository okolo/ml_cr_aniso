#!/bin/bash
PATH2CNN=$1 # should contain '/' at the end
declare -A sources
# dictionary only works in bash ver4+
sources["NGC253"]="3.5"
sources["CenA"]="3.5"
sources["M82"]="3.5"
sources["M87"]="18.5"
sources["FornaxA"]="20.0"

CNN_SUFFIX="_Bjf_Ns32-1_F32_v*.h5"

# The number of samples to be used to check the CNN
Nsamples="50000"
Nside="32"

runit(){
	GMF=$1
	Neecr=$2
	CNN_PREFIX=$3
	SRC=$4
	D=${sources[${s}]}
	CNN=${PATH2CNN}${CNN_PREFIX}${Neecr}${CNN_SUFFIX}


	# Sample for comparison:
	CMP="data/$GMF/sources/src_sample_"${SRC}"_D"${D}"_Emin56_N10000_R1_Nside32.txt.xz"

	python calc_min_fractions.py --log_sample \
				--Neecr ${Neecr} \
				--n_samples ${Nsamples} \
				--Nside ${Nside} \
				--model ${CNN} \
				${CMP} #  2>&1 >/dev/null
			}

for s in ${!sources[@]}; do
    for GMF in "jf" "jf_pl" "jf_sol" "tf" "pt" ; do
        for N in 50 100 200 300 400 500; do
            for CNN_PREFIX in "CenA_FornaxA_M87_N" "CenA_FornaxA_M82_M87_NGC253_N" ; do
                runit $GMF $N $CNN_PREFIX $s
            done
            # CNN_PREFIX=${s}_N
            # runit $GMF $N $CNN_PREFIX $s
        done
    done
done

echo "Really everything done!"

