#!/bin/bash
#SBATCH --job-name=flow_analysis
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=64
#SBATCH --time=12:00:00
#SBATCH --exclude=jpbo-002-17

module purge && module load Stages/2025 && module load GCC OpenMPI CUDA  CMake HDF5 Python Ninja ADIOS2 FFTW && source /e/project1/jureap19/venv/jupiter/bin/activate

srun python3 -u run_analysis.py \
  --box_length=1 \
  --res=2048 \
  --type=transfer \
  --terms H UU BB UBT \
  --data_type=AthenaPK_rst \
  --data_path="/e/scratch/jureap19/lenard/2k_fully_hel/parthenon.restart.00002.rhdf" \
  --eos=isothermal \
  --outfile="transfer_analysis_helical__t2" \
  --binning="log" -b
