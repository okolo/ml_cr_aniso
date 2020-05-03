# Identifying nearby sources of ultra-high-energy cosmic rays with deep learning
by Oleg Kalashev, Maxim Pshirkov, Mikhail Zotov

Please cite the following <a href="https://arxiv.org/abs/1912.00625">paper</a>
## source code and suplmental materials

### 1. Unpack KKOS spectra files

<pre><code>tar xfoj spectra_1s.tbz2 </code></pre>

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
* data/jf/32 shiuld contain deflection maps for all pair E,Z (e.g. helium_141EeV.txt.xz etc.), where jf is magnetic field model

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

