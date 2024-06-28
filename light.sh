#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/student_20/$OPTIM/"
EPOCH=20
MODE='student'
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $MODE


