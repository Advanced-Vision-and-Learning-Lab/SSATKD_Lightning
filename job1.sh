#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/CNN14_struct4/$OPTIM/"
FOLDER2="Saved_Models/ResNet38KD_struct4/$OPTIM/"
FOLDER3="Saved_Models/MobileNetV1_struct4/$OPTIM/"
EPOCH=150
mode='distillation'
teacher='CNN_14'
teacher2='ResNet38'
teacher3='MobileNetV1'
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --mode $mode --teacher_model $teacher
python demo.py --folder $FOLDER2 --optimizer $OPTIM --num_epochs $EPOCH --mode $mode --teacher_model $teacher2
python demo.py --folder $FOLDER3 --optimizer $OPTIM --num_epochs $EPOCH --mode $mode --teacher_model $teacher3

#11.44
