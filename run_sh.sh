#This is the file that we have been using to run code in the cluste

#!/bin/bash
#SBATCH -n 2 # Number of cores
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -D /fhome/pmlai09 # working directory
#SBATCH -p tfg # Partition to submit to
#SBATCH --mem 2048 # 2GB solicitados.
#SBATCH -o error_folder/%x_%u_%j.out # File to which STDOUT will be written
#SBATCH -e error_folder/%x_%u_%j.err # File to which STDERR will be written
#SBATCH --gres gpu:1 # Para pedir graficas

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #Environment variable in the error message of job 51765 (la que petó la ram a los 228 episodios)
python3 /fhome/pmlai09/dqn3.py $1

