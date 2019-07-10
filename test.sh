#!/bin/bash --login
########## BATCH Lines for Resource Request ##########

#SBATCH --time=2:00:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=8                   # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=8                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=10G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --constraint=intel18
#SBATCH --job-name test_method_b      # you can give your job a name for easier identification (same as -J)

########## Command Lines to Run ##########

module purge
module load intel/2018a FFTW   ### load necessary modules, e.g.
export PATH=/mnt/home/f0008575/yt-conda/bin:$PATH

echo "Beginning analysis"

srun -n 8 --mem-per-cpu=20G --time=00:20:00 python ~/src/energy-transfer-analysis/run_analysis.py --res 256 --data_path /mnt/scratch/f0008575/a-1.00/id0/Turb.0010.vtk  --data_type Athena --type flow --eos isothermal -forced --outfile /mnt/scratch/f0008575/a-1.00/method_b_0010.hdf5 --kernel KernelGauss

echo "Finished analysis"

