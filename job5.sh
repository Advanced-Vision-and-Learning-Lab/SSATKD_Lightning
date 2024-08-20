#!/bin/bash

# Set variables
OPTIM="AdamW"
FOLDER="Saved_Models/HLTDNNLogMel/$OPTIM/"
EPOCH=150
mode='student'
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $mode 


