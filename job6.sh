#!/bin/bash

# Set variables
OPTIM="Adagrad"
FOLDER="Saved_Models/random_test/$OPTIM/"
EPOCH=50
mode='distillation'
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $mode 

