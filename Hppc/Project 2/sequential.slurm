#!/usr/bin/env bash
#SBATCH --ntasks=1
#SBATCH --output=stdout.txt

singularity exec ~/modi_images/hpc-notebook-latest.simg ./fwc_sequentail --iter 50 --model world_models/small.hdf5
