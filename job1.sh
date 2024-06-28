#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/new_losses/$OPTIM/"
EPOCH=50
MODE='distillation'



python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $MODE

