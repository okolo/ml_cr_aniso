#!/usr/bin/env bash

model_dir=$1

score_name=best_val_frac_pt

for N in 50 100 200 300 400 500
do
    for source in M82 CenA NGC253 M87 FornaxA
    do
        prefix=${model_dir}/${source}_N${N}
        f=`for file in ${prefix}_Bjf_Ns32-1_F32_v*.h5.score ; do cat ${file} | grep ${score_name} | awk '{print file " " $2}' file=$file ; done | sort -g -k 2 | awk 'NR==1{print $1}' | sed 's/.score//'`
        ln -s $f .
    done
done

