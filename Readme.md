# Energy transfer analysis

## Requirements
 - mpiFFT4py package (incl dependencies)
 - yt package

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
- FluidType (dtype string - e.g. mhd) currently supported options are
  - "mhd"
  - "hydro" 
- Binning (dtype string - e.g. testing) to determine the bin widts. Currently supported
  - "Lin" leads to linearly equally spaced bins with boundaries at $k = 0.5,1.5,2.5,...,Res/2$
  - "Log" leads to logarithmically equally spaced bins with boundaries at $k = 0, 4 * 2^{(i - 1)/4},Res/2$
  - "testing" leads to Bins used for regression testing, i.e. $k = 0.5,1.5,2.5,16.0,26.5,28.5,32.0$


## Current limitations
- data can only be read via an existing yt frontend
- data is assumed to be in a periodic box with side length 1 and equal grid spacing
- pressure is calculated assuming an isothermal EOS with $c_s = 1$
- wavenumber are implicitly normalized (k = 1, ...)
- only slab decomposition (of equal size) among MPI processes 
- units are hard coded/implicitly assumes
 - velocity has units of speed of sound (with $c_s = 1$)
 - magnetic field includes $1/\sqrt{4 \pi}$

## Adding new simulation output
- Edit `runTransfer.py` and add another `SimType` around line 50.
- The field variables are the strings that are availble for that particular dump within it, i.e.
they should be present in the `ds.field_list` array.
- The `loadPath` variable should be formated in the same way yt would load the datasta. It is eventually used in `yt.load(loadPath)`

