#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=Job               # Set the job name to Job
#SBATCH --time=12:00:00              # Set the wall clock limit to HH:MM:SS
#SBATCH --nodes=1                    # How many nodes to request, this should almost always be 1
#SBATCH --cpus-per-task=1            # How many cpus to request per task, when using with lightning this should be > --num_data_workers hyperparameters
#SBATCH --mem=16G                    # Request RAM that is per node 8 to 16GB is usually sufficient, this depends on the size of the dataset
#SBATCH --output=kd.%j.out           # Redirect stdout to file
#SBATCH --error=kd.%j.err            # Redirect stderr to file
#SBATCH --partition=gpu              # Specify partition to submit job to, dont change this
#SBATCH --gres=gpu:2              # Specify GPU(s) per node, 2 A100 gpu; Use more than 1 if you are running out of memory due to the GPU


# source activate lightkd

# Set variables
OPTIM="Adagrad"
FOLDER="Saved_Models/TDNN/$OPTIM/"
EPOCH=300
MODE="student"

# First Executable Line
python3 demo.py --folder $FOLDER --num_epochs $EPOCH --mode $MODE --teacher_model $TEACHER


