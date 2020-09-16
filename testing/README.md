# Regression testing

## Manual testing

Gold standard is `0024-All-Forc-Pres-testing-128_HDF-gold.pkl`

It is based on a snapshot of a driven turbulence simulation
run with Enzo at $128^3$, see run/MHD/3D/StochasticForcing folder in Enzo. 
Snapshot is available at: https://pgrete.de/dl/EnTrans/EnTransTesting-DD0024.tgz (370M)
```
md5
9b2dfb5412770ae63277726de39b8f63  EnTransTesting-DD0024.tgz
sha1
d3df730b08943a2067f377140943f70ff8b9fd20  EnTransTesting-DD0024.tgz
```
The `DD0024` folder in the archive contains
- `StochasticForcing.enzo`: the Enzo parameter file
- the orginal Enzo data dump
- `*-128.hdf5`: the extracted data dump in HDF5 format (column major ordering) at original resolution
- `*-64.hdf5`: the extracted data dump in HDF5 format (column major ordering) at half resolution (using volume averages)
- `shrink.py`: the script that was used to create the HDF5 data

Bin boundaries used for the transfer analysis: `Bins = [0.5,1.5,2.5,16.0,26.5,28.5,32.0]`

To run test simply 
```
mpirun -np 8 python ../run_analysis.py --terms All FU PU BUPbb UBPbb --res 128 --data_path DD0024/data0024 --data_type Enzo --binning test --type transfer --outfile test.pkl --eos adiabatic --gamma 1.0001  -forced -b
python runTest.py 0024-All-Forc-Pres-testing-128_HDF-gold.pkl test.pkl
```

## Automated testing

The tests above are run automatically for each commit through the [GitLab mirror](https://gitlab.com/pgrete/energy-transfer-analysis).

The CI configuration is set in [.gitlab-ci.yml](../.gitlab-ci.yml) and the corresponding
Docker image is created from the [Dockerfile](Dockerfile).
