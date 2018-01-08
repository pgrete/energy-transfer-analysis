# Changelog
- Exchanged mpiFFT4py with mpi4py-fft (cleaner, simpler interface by same group)
- Support for adiabatic EOS (with gamma = 5/3 hardcoded at the moment)
- Added script for higher order turbulent flow analysis, see below
- Data in column major format stored in HDF5 files is only read by proc 0 and then distributed. Data stored in row major format is read in parallel by all processes.

# Requirements
- mpi4py-fft  package (incl dependencies) [homepage/download](https://bitbucket.org/mpi4py/mpi4py-fft)
- yt package (optional - can be used to read initial data)

# Energy transfer analysis

## Usage

The `runTransfer.py` script does all the job. There are several command line options (all required).
For example,
```
mpirun -np 8 python runTransfer.py 0024 All-Forc-Pres 128 Enzo mhd testing
```
Following six parameters need to be set (in that order)
- ID (dtype int - e.g. 0024) of the data dump, also used as an identified for output files
- Terms (dtype string - e.g. All-Forc-Pres) separated by "-". Currently supported options are
  - "All" includes UU, BB, BUT, BUP, UBT, UBP for MHD and UU for hydro
  - "Pres" includes PU pressure term
  - "Forc" includes FU force term
- Res (dtype int - e.g. 128) linear resolution of snapshot
- SimType (dtype string - e.g. Enzo) to internally determine the field names yt should load (see section "Adding new simulation output"
- FluidType (dtype string - e.g. mhd) that may include the keyword 'adiabatic'. By default an isothermal EOS is assumed with fixed soundspeed of 1 (code units), so that pressure = density. Examples are
  - "mhd"
  - "hydro" 
  - "mhd-adiabatic"
  - "hydro-adiabatic"
- Binning (dtype string - e.g. testing) to determine the bin widts. Currently supported
  - "Lin" leads to linearly equally spaced bins with boundaries at $k = 0.5,1.5,2.5,...,Res/2$
  - "Log" leads to logarithmically equally spaced bins with boundaries at $k = 0, 4 * 2^{(i - 1)/4},Res/2$
  - "testing" leads to Bins used for regression testing, i.e. $k = 0.5,1.5,2.5,16.0,26.5,28.5,32.0$


## Current limitations
- data is assumed to be in a periodic box with side length 1 and equal grid spacing
- pressure is calculated with $c_s = 1$ in the isothermal EOS case
- wavenumber are implicitly normalized (k = 1, ...)
- only slab decomposition (of equal size) among MPI processes 
- units are hard coded/implicitly assumes
 - velocity has units of speed of sound (with $c_s = 1$)
 - magnetic field includes $1/\sqrt{4 \pi}$

## Adding new simulation output
- Edit `runTransfer.py` and add another `SimType` around line 50.
- The field variables are the strings that are availble for that particular dump within it, i.e.
they should be present in the `ds.field_list` array.
- The `loadPath` variable should be formated in the same way yt would load the dataset. It is eventually used in `yt.load(loadPath)`

# Higher order turbulent flow analysis

## Features
- Uses MPI with slab 
- For a given scalar field 
  - higher order statistical moments for arb. fields incl. mean, rms, variance, standard deviation, skewness, kurtosis, minimum, maximum, absolute minimum, absolute maximum
  - 1d histograms with automatic and given bounds 
- For two scalar field
  - correlation coefficient
  - 2d histograms with automatic and given bounds
- Decomposion of vector fields in harmonic, solenoidal and compressive modes
- Power spectra
  - for total, solenoidal and compressive components
  - with different normalization: no weighting, surface average, shell average
- Dispersion measures, rotation measures and line of sight magnetic field along all axes
- see also plotHigherOrder.ipynb

## Usage
The `runHigherFlowQuant.py` script does all the job. There are several command line options (all required) similar to the energy transfer script (a lot in the beginning should actually be merged).
For example,
```
mpirun -np 8 python runHigherFlowQuant.py 0024 128 AthenaHDF mhd-adiabatic forced
```
Following five parameters need to be set (in that order)
- ID (dtype int - e.g. 0024) of the data dump, also used as an identified for output files
- Res (dtype int - e.g. 128) linear resolution of snapshot
- SimType (dtype string - e.g. Enzo) to internally determine the field names yt should load (see section "Adding new simulation output" or which HDF files should be read.
- FluidType (dtype string - e.g. mhd) that may include the keyword 'adiabatic'. By default an isothermal EOS is assumed with fixed soundspeed of 1 (code units), so that pressure = density. Examples are
  - "mhd"
  - "hydro" 
  - "mhd-adiabatic"
  - "hydro-adiabatic"
- TurbType (dtype string). Determines whether an acceleration field is present. Options are
  - "forced" (acceleration field is read and included in statistics)
  - "decay" (no acceleration field)

## Current limitations
- Most content in a single, mostly undocumented file...
- Limits for global binning are hardcoded in the main file
- Uniform, static grids with assumed box size of L = 1
