#!/bin/sh
### Note: No commands may be executed until after the #PBS lines
### Account information
#PBS -W group_list=ht3_aim -A ht3_aim
### Job name (comment out the next line to get the name of the script used as the job name)
#PBS -N embeddings_prottrans
### Output files (comment out the next 2 lines to get the job name used instead)
#PBS -e errors/embeddings_prottrans.err
#PBS -o errors/embeddings_prottrans.log
### Only send mail when job is aborted or terminates abnormally
### Number of nodes
#PBS -l nodes=1:ppn=4:gpus=1
### Memory
#PBS -l mem=32gb
### Requesting time - format is <days>:<hours>:<minutes>:<seconds>
#PBS -l walltime=8:00:00
  
# Go to the directory from where the job was submitted (initial directory is $HOME)
echo Working directory is $PBS_O_WORKDIR
cd $PBS_O_WORKDIR

# Load all required modules for the job
module load tools
module load anaconda3/4.4.0

export LC_ALL=en_US.utf-8
export LANG=en_US.utf-8

#pip install papermill

papermill embeddings_transprot.ipynb embeddings_transprot.ipynb
