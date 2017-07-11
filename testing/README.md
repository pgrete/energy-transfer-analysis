# Regression testing

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
python runTest.py 0024-All-Forc-Pres-testing-128_HDF-gold.pkl <testFile>
```
