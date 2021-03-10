#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=ht3_aim -A ht3_aim
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N embeddings_transprot
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e /home/projects/ht3_aim/people/erikie/nsp/notebooks/errors/embeddings_transprot.err
#PBS -o /home/projects/ht3_aim/people/erikie/nsp/notebooks/errors/embeddings_transprot.log
### Only send mail when job is aborted or terminates abnormally
### Number of nodes
#PBS -l nodes=1:ppn=4:gpus=1
### Memory
#PBS -l mem=32gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds> (here, 12 hours)
#PBS -l walltime=12:00:00
  
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Load all required modules for the job
module load cuda/toolkit/10.1/10.1.168
module load tools
module load anaconda3/4.4.0

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

# ensure papermill is installed
#pip install papermill

pip install transformers

papermill embeddings_transprot.ipynb embeddings_transprot.ipynb
