#Baseline TDNN Experiments (Run experiments and get results)


OPTIM="Adagrad"
FOLDER="Saved_Models/loss_curve/$OPTIM/"
EPOCH=30
MODE='distillation'


#python demo.py --audio_feature Mel_Spectrogram --folder $FOLDER --histogram --optimizer $OPTIM --num_epochs $EPOCH 
#python View_Results.py --audio_feature Mel_Spectrogram --folder $FOLDER --histogram --optimizer $OPTIM 
#python demo.py --audio_feature MFCC --folder $FOLDER --histogram --optimizer $OPTIM --num_epochs $EPOCH 
#python View_Results.py --audio_feature MFCC --folder $FOLDER --histogram --optimizer $OPTIM 
python demo.py --folder $FOLDER --optimizer $OPTIM --num_epochs $EPOCH --no-feature_extraction --mode $MODE
python View_Results.py --folder $FOLDER --optimizer $OPTIM --feature_extraction
#python demo.py --audio_feature GFCC --folder $FOLDER --histogram --optimizer $OPTIM --num_epochs $EPOCH 
#python View_Results.py --audio_feature GFCC --folder $FOLDER --histogram --optimizer $OPTIM 
#python demo.py --audio_feature CQT --folder $FOLDER --model $MODEL --teacher_model $TEACHER --no-histogram --optimizer $OPTIM --num_epochs $EPOCH 
#python View_Results.py --audio_feature CQT --folder $FOLDER --model $MODEL --teacher_model $TEACHER --no-histogram --optimizer $OPTIM 
#python demo.py --audio_feature VQT --folder $FOLDER --histogram --optimizer $OPTIM --num_epochs $EPOCH 
#python View_Results.py --audio_feature VQT --folder $FOLDER --histogram --optimizer $OPTIM 
