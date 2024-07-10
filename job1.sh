#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/CNN_3_Runs/$OPTIM/"
EPOCH=300
mode='teacher'
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $mode

