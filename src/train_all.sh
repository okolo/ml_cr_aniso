#!/usr/bin/env bash
n_samples=10000
min_version=8
monitor=frac_compare
epochs=50
for N in 50 100 200 300 400 500
do
    for source in M82 CenA NGC253 M87 FornaxA
    do
        python train_healpix.py --n_epochs $epochs --min_version $min_version --source_id $source --Neecr $N --log_sample --n_samples $n_samples --mf jf --n_early_stop 2 --Nside 32 --Nini 10000 --deterministic --monitor $monitor
    done
    python train_healpix.py --n_epochs $epochs --min_version $min_version --source_id "CenA,FornaxA,M87" --Neecr $N --log_sample --n_samples $n_samples --mf jf --n_early_stop 2 --Nside 32 --Nini 10000 --deterministic --monitor $monitor
done
for N in 100 200 300 400 500
do
   python train_healpix.py --n_epochs $epochs --min_version $min_version --source_id all --Neecr $N --log_sample --n_samples $n_samples --mf jf --n_early_stop 2 --Nside 32 --Nini 10000 --deterministic --monitor $monitor
done
