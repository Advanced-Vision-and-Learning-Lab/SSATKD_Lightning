#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/AdamW0.001CNN14_3runs/$OPTIM/"
EPOCH=50
mode='teacher'
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $mode

