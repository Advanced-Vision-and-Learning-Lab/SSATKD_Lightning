#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/student_split/$OPTIM/"
EPOCH=100
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH 


