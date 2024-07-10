#!/bin/bash

# Set variables
OPTIM="Adagrad"
FOLDER="Saved_Models/Res1dNet31_300/$OPTIM/"
EPOCH=300
MODE="teacher"
TEACHER="Res1dNet31"

# First Executable Line
python3 demo.py --folder $FOLDER --num_epochs $EPOCH --mode $MODE --teacher_model $TEACHER

