# Changelog
- Exchanged mpiFFT4py with mpi4py-fft (cleaner, simpler interface by same group)
- Support for adiabatic EOS (with gamma = 5/3 hardcoded at the moment)
- Added script for higher order turbulent flow analysis, see below
- Data in column major format stored in HDF5 files is only read by proc 0 and then distributed. Data stored in row major format is read in parallel by all processes.

# Requirements
- mpi4py-fft  package (incl dependencies) [homepage/download](https://bitbucket.org/mpi4py/mpi4py-fft)
- h5py package
- yt package (optional - can be used to read initial data)
- palettable package (optional - used for colors in the sample plotting notebook)

# General usage

The `run_analysis.py` script is the main file that needs to be run.

In general the following parameters are available
- `--res RES`             set linear resolution of the cubic box
- `--type {transfer,flow,unit-test}`  set analysis type
  - `transfer` for energy transfer analysis
  - `flow` for analysis of turbulence statistics
  - `unit-test` for some preliminary unit tests
- `--data_type {Enzo,AthenaPP,AthenaHDFC,Athena,Flash}` set data cube type
  - `Enzo` reads Enzo data using `yt` frontend
  - `AthenaPP` reads Athena++/K-Athena data using `yt` frontend
  - `AthenaHDFC` reads Athena data that has been converted to hdf5 data
  - `Athena` reads Athena data using `yt` frontend
  - `Flash` read Flash data using `yt` frontend
  - to add more options see section "Adding new simulation output"
- `--data_path DATA_PATH` set data location
-  `--outfile OUTFILE`     set file to store results (should be `.pkl` for `transfer` and `.hdf5` for `flow`)
-  `--extrema_file EXTREMA_FILE` Path to pickled python dictionary containing minimum and maximum values for quantities (used for creating histograms with a fixed [global] bounds)
   - Style of dictionary is for example. `{'rho' : [ 0., 10]}`
   - All quantities are always binned to the min and max values of the individual snapshot.
   - If no dictionary is found under the given path (e.g., by setting it to a nonexisting file/path) no histograms with global bounds will be created
-  `-b`                    enable magnetic fields
-  `-forced`               output is actively forced
-  `--eos {isothermal,adiabatic}` set equation of state
-  `--gamma GAMMA`         set adiabatic gamma index
-   `-approx-isothermal`    assume c_s^2 / gamma = p/rho = 1
-   `--terms {All,Int,UU,BUT,BUP,UBT,UBPb,BB,BUPbb,UBPbb,SS,SU,US,PU,FU}` set energy transfer terms to analyze
-  `--binning {log,lin,test}`  set binning used in energy transfer analysis
   - `lin` leads to linearly equally spaced bins with boundaries at $k = 0.5,1.5,2.5,...,Res/2$
   - `log` leads to logarithmically equally spaced bins with boundaries at $k = 0, 4 * 2^{(i - 1)/4},Res/2$
   - `test` leads to bins used for regression testing, i.e. $k = 0.5,1.5,2.5,16.0,26.5,28.5,32.0$
-  `--kernels` choose one or more real space convolution kernels to be used in filtering
   - `Box` for a box car/top hat filter (implementation needs update)
   - `Sharp` for a sharp spectral filter
   - `Gauss` for a smooth Gaussian filter


## Energy transfer analysis

### Sample usage

Use the `run_analysis.py` script with the `--flow transfer` option.
For example (to run the transfer analysis on the regression data set),
```
srun -n 8 python ./run_analysis.py --terms All FU PU BUPbb UBPbb --res 128 --data_path DD0024/data0024 --data_type Enzo --binning test --type transfer --outfile test-out.pkl --eos adiabatic --gamma 1.0001  -forced -b
```

## Turbulent flow analysis
### Features
- Uses MPI with slab decomposition
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
  - with different definitions of kinetic energy density
    - $E(k) = \sqrt(\rho u)^2$ (Grete, et al., 2017)
    - $E(k) = |\overline{\rho u_l}|^2 / 2\rho_l$ (Sadek & Aluie, 2018)
       - must specify convolution kernel type with `--kernels`
- Dispersion measures, rotation measures and line of sight magnetic field along all axes

## Usage
Use the `run_analysis.py` script with the `--flow flow` option.
For example (to analyze a driven turbulence hydro simulation with an isothermal equation of state),
For example,
```
srun -n 8 python ./run_analysis.py --res 256 --data_path /PATH/TO/SIM/DUMP  --data_type Athena --type flow --eos isothermal --outfile /PATH/TO/OUTFILE.hdf5 -forced --kernels Gauss
```

### Sample analysis
The notebook `sample_flow_analysis.ipynb` contains examples on how to analyze/visualize/plot the data from the flow analysis.

# Current limitations
- data is assumed to be in a periodic box with side length 1 and equal grid spacing
- pressure is calculated with $c_s = 1$ in the isothermal EOS case
- wavenumber are implicitly normalized (k = 1, ...)
- only slab decomposition (of equal size) among MPI processes 
- units are hard coded/implicitly assumes
 - velocity has units of speed of sound (with $c_s = 1$)
 - magnetic field includes $1/\sqrt{4 \pi}$

# Adding new simulation output
- Edit `run_analysis.py` and add another option to the `--data_type` argument
- Edit `IOhelperFuncs.py` and make sure that the field variables are the strings
that are availble for that particular dump within it, i.e.
they should be present in the `ds.field_list` array.


