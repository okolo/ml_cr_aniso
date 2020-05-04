# Identifying nearby sources of ultra-high-energy cosmic rays with deep learning
by Oleg Kalashev, Maxim Pshirkov, Mikhail Zotov

Please cite the following <a href="https://arxiv.org/abs/1912.00625">paper</a>
## source code and suplmental materials

### 1. Unpack KKOS spectra files

    cd src
    tar xfoj data/spectra_1s.tbz2

### 2. Sample E,Z pairs

1. Sample 100000 events (E,Z pairs) with energy above 56 EeV for the source located at 3.5 Mpc 

    <pre><code>python psample.py --distance 3.5 --Nini 100000 --Emin 56</code></pre>

    * same with mass composition shift A -> A/2

    <pre><code>python psample.py --distance 3.5 --Nini 100000 --Emin 56 --shiftA 0.5</code></pre>
   
    * same with monochromatic composition, e.g. A=16

    <pre><code>python psample.py --distance 3.5 --Nini 100000 --Emin 56 --shiftA -16</code></pre>

2. sorting output

<pre><code>cat sample_D3.5_Emin56_100000nuclei.txt | awk 'NR>6' | sort | uniq -c | awk '{print $2 " " $3 " " $1}' > data/sample_D3.5_Emin56_100000nuclei_sorted.txt</code></pre>

### 3. Deflection map preparation

#### Prerequisites:
Install <a href="https://github.com/CRPropa/CRPropa3">crpropa</a> along with python integration
#### Command
To calculate deflection map for pair _E, Z_ run command

<pre><code>python galback.py Z E</code></pre>

You can edit healpix grid resolution parameter Nside and galactic magnetic fiels model in galback.py
For list of available galactic magnetic field models scroll to line
<pre><code># Model of the Galactic Magnetic Field</code></pre>
in _galback.py_


### 4. From-source event map generation
#### Prerequisites:

* data/sample_D3.5_Emin56_100000nuclei_sorted.txt generated in step 2.2 (here 3.5 is distance to CenA)
* data/jf/32 should contain deflection maps for all pair E,Z (e.g. helium_141EeV.txt.xz etc.), where jf is magnetic field model

#### Command
<pre><code>python src_sample.py --source_id CenA --Nside 32 --Nini 100000 --GMF jf --Emin 56</code></pre>

#### Output
file with arrival directions density map
_data/jf/sources/src_sample_CenA_D3.5_Emin56_N100000_R1_Nside32.txt.xz_

  
### 5. Train classifier on HEALPix grid and estimate minimal from-source event fraction

#### Prerequisites:
* Corresponding from-source event maps prepared in step 4. should be located in folder
_data/jf/sources/_
#### Command

For single source classifier

    python train_healpix.py --source_id CenA --Neecr 500 --log_sample --n_samples 100000 --mf jf --n_early_stop 2 --Nside 32

For multisource classifier

    python train_healpix.py --source_id "CenA,M82,NGC253,M87,FornaxA" ...

To check the resulting test statistic performance for alternative magnetic field model
use --compare_mf flag:

    python train_healpix.py --mf jf --compare_mf pt ...

#### Output

* CenA_N500_Bjf_Ns32-1_F32_v0.h5 neural network classifier 
* CenA_N500_Bjf_Ns32-1_F32_v0.h5.score text file with classifier and test statistic metrics
* CenA_N500_Bjf_Ns32-1_F32_v0_det_frac.txt log file containing the evolution of minimal from-source event fraction 
during neural network training

### 6. Apply pretrained network classifier to an arbitrary event map

#### Prerequisites:
* Event map(s) must be saved in _src_sample_ format (see step 4.)
* To be able to load the models saved earlier to _.h5_ apply patch to NNhealpix library
<pre><code>cd (NNhealpix dir)/nnhealpix/layers
patch < (uhecr_aniso dir)/src/nnhealpix_layers.patch
</code></pre>

#### Command
    python calc_min_fractions.py data/jf/sources/src_sample_CenA_D3.5_Emin56_N10000_R1_Nside32_shift2.0.txt.xz --log_sample --Neecr 300 --n_samples 10000 --Nside 32 --model CenA_FornaxA_M82_M87_NGC253_N300_Bjf_Ns32-1_F32_v0.h5

In case several _src_sample_ files are given, maps are generated using each of them
in roughly equal ammounts (only one random map is used for each sample generation)

With flag _--fractions_ several maps in given proportions can be used for each sample generation, e.g.:

    python calc_min_fractions.py data/jf/sources/src_sample_CenA_D3.5_Emin56_N10000_R1_Nside32.txt.xz  data/jf/sources/src_sample_FornaxA_D20.0_Emin56_N10000_R1_Nside32.txt.xz  --fractions 3 1 --log_sample --Neecr 300 --n_samples 10000 --Nside 32 --model CenA_FornaxA_M82_M87_NGC253_N300_Bjf_Ns32-1_F32_v0.h5

#### Output
Program outputs minimal detectable from-source event fraction on particular map(s) along with type I error alpha.
The information is appended to log file
CenA_FornaxA_M82_M87_NGC253_N300_Bjf_Ns32-1_F32_v0.h5_cmp.txt

## Angular power spectrum based test statistic

### 1. Generate sample data files

Two groups of sample data files should be generated - training and testing. Each group should contain at least one with
mixed sample spectra and at least one with isotropic data

#### mixed sample datafile generation:

    python3 mixed_spectrum_gen.py --Neecr 50 --Nmixed_samples 100000 --source_id CenA --log_sample --f_src_min 1e-2

Output is saved to aps_CenA_D3.5_Bjf_Emin56_Neecr50_Nsample100000_R1_Nside512_logF_src_f_min0.01_0.npz
Consequent multiple executions of the above command in the same folder will produce files xxx_1.npz, xxx_2.npz,
etc. with unique samples which is ensured by random seed initialization 

#### isotropic coefficients generation:

    python3 mixed_spectrum_gen.py --Neecr 50 --Nmixed_samples 100000 --source_id CenA --f_src 0

Output is saved to iso_Neecr50_Nsample100000_Nside512_0.npz, iso_Neecr50_Nsample100000_Nside512_1.npz, etc.

#### select file used for normalization

This could be any file created on step 1. Edit train_spec.py to set _norm_file_ parameter

    norm_file = 'aps_CenA_D3.5_Emin56_Neecr500_Nsample3000_R1_Nside512_100.npz'

### 2. Train classifier

    python3 train_spec.py aps_CenA_D3.5_Bjf_Emin56_Neecr50_Nsample100000_R1_Nside512_logF_src_f_min0.01_0.npz iso_Neecr50_Nsample100000_Nside512_0.npz

Here use files created in previous step as parameters.
#### Output
spectrum_L33_th0.01_v0.h5 trained classifier

### 3. Calculate test statistic on test data
#### Prerequisites:
* Two or more files generated in step 1 which belong to the testing group

#### Command:

    python3 nn_f_spec.py aps_CenA_D3.5_Bjf_Emin56_Neecr50_Nsample100000_R1_Nside512_logF_src_f_min0.01_1.npz iso_Neecr50_Nsample100000_Nside512_1.npz --model spectrum_L33_th0.01_v0.h5

#### Output
Files containing test statistics defined by the classifier, calculated on the samples provided
spectrum_L33_th0.01_v0__aps_CenA_D3.5_Bjf_Emin56_Neecr50_Nsample100000_R1_Nside512_logF_src_f_min0.01_1.npz
spectrum_L33_th0.01_v0__iso_Neecr50_Nsample100000_Nside512_1.npz

### 4. Calculate minimal detectable fractions 

#### Prerequisites:
* Files containing test statistics calculated in step 3

#### Command

    python3 calc_fractions_ps.py --mixed
spectrum_L33_th0.01_v0__aps_CenA_D3.5_Bjf_Emin56_Neecr50_Nsample100000_R1_Nside512_logF_src_f_min0.01_1.npz --iso spectrum_L33_th0.01_v0__iso_Neecr50_Nsample100000_Nside512_1.npz

The command output to the console:
detectable_fraction, alpha (type I error probability)

### 5 Arbitrary test statistic evaluation

One could use arbitrary test statistics instead of neural network based classifier. To do this edit _f_spec.py_
to define your custom statistic 

    def f(spec):
        ts = ..# define your statistic here
        return ts

and replace step 3 by the following command

    python3 f_spec.py aps_CenA_D3.5_Bjf_Emin56_Neecr50_Nsample100000_R1_Nside512_logF_src_f_min0.01_1.npz iso_Neecr50_Nsample100000_Nside512_1.npz

