#!/bin/bash

# Set variables
OPTIM="Adagrad"
FOLDER="Saved_Models/student_shuffle/$OPTIM/"
EPOCH=300
MODE="student"



# First Executable Line
python3 demo.py --folder $FOLDER --num_epochs $EPOCH --mode $MODE

