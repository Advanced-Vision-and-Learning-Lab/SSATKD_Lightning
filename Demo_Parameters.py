
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 11:20:44 2019
Parameters for histogram layer experiments
Only change parameters in this file before running
demo.py
@author: jpeeples 
"""

def Parameters(args):
    ######## ONLY CHANGE PARAMETERS BELOW ########
    #Flag for if results are to be saved out
    #Set to True to save results out and False to not save results
    save_results = args.save_results
    
    #Location to store trained models
    #Always add slash (/) after folder name
    folder = args.folder
    
    #optimizer selection
    optimizer = args.optimizer
    #loss = args.loss
    
    #Flag to use histogram model or baseline global average pooling (GAP)
    # Set to True to use histogram layer and False to use GAP model
    histogram = args.histogram
    
    #Sigma value for synthetic dataset
    #sigma = args.sigma
    
    #Select dataset. Set to number of desired texture dataset
    # For KTH, currently training on 2 samples, validating on 1 sample, and testing
    # on 1 sample
    data_selection = args.data_selection
    Dataset_names = {0: 'DeepShip'}
    HPRC = args.HPRC


    
    #Number of bins for histogram layer. Recommended values are 4, 8 and 16.
    #Set number of bins to powers of 2 (e.g., 2, 4, 8, etc.)
    #For HistRes_B models using ResNet18 and ResNet50, do not set number of bins
    #higher than 128 and 512 respectively. Note: a 1x1xK convolution is used to
    #downsample feature maps before binning process. If the bin values are set
    #higher than 128 or 512 for HistRes_B models using ResNet18 or ResNet50
    #respectively, than an error will occur due to attempting to reduce the number of
    #features maps to values less than one
    numBins = args.numBins
    
    
    #Flag for feature extraction. False, train whole model. True, only update
    #fully connected and histogram layers parameters (default: False)
    #Flag to use pretrained model from ImageNet or train from scratch (default: True)
    #Flag to add BN to convolutional features (default:True)
    #Location/Scale at which to apply histogram layer (default: 5 (at the end))
    feature_extraction = args.feature_extraction
    use_pretrained = args.use_pretrained
    add_bn = True
    scale = 5
    
    #Set learning rate for new layers
    #Recommended values are .01 (used in paper) or .001
    lr = args.lr
    
    #Parameters of Histogram Layer
    #For no padding, set 0. If padding is desired,
    #enter amount of zero padding to add to each side of image
    #(did not use padding in paper, recommended value is 0 for padding)
    padding = 0
    

    #Reduce dimensionality based on number of output feature maps from GAP layer
    #Used to compute number of features from histogram layer
    out_channels = {"resnet50": 2048, "resnet18": 512, "efficientnet": 1280, "CustomResNet18WithCDM":512,
                    "resnet50_wide": 2048, "resnet50_next": 2048, "densenet121": 4096,
                    "regnet": 400, "TDNN": 256,"DNN":256,'modified_resnet50':2048,'CDM':256,'DTIEM':256,'CustomResNet50WideWithCDM':2048,"CNN_14": 2048}
    
    #Set whether to have the histogram layer inline or parallel (default: parallel)
    #Set whether to use sum (unnormalized count) or average pooling (normalized count)
    # (default: average pooling)
    #Set whether to enforce sum to one constraint across bins (default: True)
    parallel = True
    normalize_count = True
    normalize_bins = True
    
    #Set step_size and decay rate for scheduler
    #In paper, learning rate was decayed factor of .1 every ten epochs (recommended)
    step_size = 10
    gamma = .1
    
    #Batch size for training and epochs. If running experiments on single GPU (e.g., 2080ti),
    #training batch size is recommended to be 64. If using at least two GPUs,
    #the recommended training batch size is 128 (as done in paper)
    #May need to reduce batch size if CUDA out of memory issue occurs
    batch_size = {'train': args.train_batch_size, 'val': args.val_batch_size, 'test': args.test_batch_size} #128
    # batch_size = {'train': 64, 'test': 64}
    num_epochs = args.num_epochs
    level_num = args.level_num
    

    
    #Pin memory for dataloader (set to True for experiments)
    pin_memory = True
    
    #Set number of workers, i.e., how many subprocesses to use for data loading.
    #Usually set to 0 or 1. Can set to more if multiple machines are used.
    #Number of workers for experiments for two GPUs was three
    # num_workers = 0
    # cpus = os.getenv('SLURM_CPUS_PER_TASK')
    num_workers = 4
    
    #Output feature map size after histogram layer
    feat_map_size = 4
    
    #Select audio feature for DeepShip 
    feature = args.audio_feature
    
    #Set filter size and stride based on scale
    # Current values will produce 2x2 local feature maps
    if scale == 1:
        stride = [32, 32]
        in_channels = {'TDNN': 16,'CNN14':64}
        kernel_size = {'TDNN': [3,3],'CNN14':[64,64]}
    elif scale == 2:
        stride = [16, 16]
        in_channels = {'CNN14':256}
        kernel_size = {'TDNN': [3,3],'CNN_14': [3,3]}
    elif scale == 3:
        stride = [8, 8]
        in_channels = {'CDM':16}
        kernel_size = {'TDNN': [3,3]}
    else:
        stride = [2, 2]
        in_channels = {'TDNN': 4,'CNN_14': 2048}
        kernel_size = {'TDNN': [3,3],'CNN_14': [3,3]}

    #Visualization of results parameters
    #Visualization parameters for figures
    fig_size = (20,20)
    font_size = 18

    #12 and 24
    #Flag for TSNE visuals, set to True to create TSNE visual of features
    #Set to false to not generate TSNE visuals
    #Separate TSNE will visualize histogram and GAP features separately
    #If set to True, TSNE of histogram and GAP features will be created
    #Number of images to view for TSNE (defaults to all training imgs unless
    #value is less than total training images).
    TSNE_visual = False
    Separate_TSNE = False
    Num_TSNE_images = 10000
    
    #Set to True if more than one GPU was used
    Parallelize_model = True
    task_flag = [True, True, True, True]
    
    ######## ONLY CHANGE PARAMETERS ABOVE ########
    if feature_extraction:
        method = 'Feature_Extraction'
    else:
        method = 'Fine_Tuning'
    if use_pretrained:
        model = 'Pretrained'
    else: 
        model = 'Scratch'
        
  
    
    
    #Location of texture datasets
    Data_dirs = {'DeepShip': './Datasets/DeepShip/'}
    segment_length = 5
    sample_rate =32000
    
    class_names = {'DeepShip':['Cargo', 'Passengership', 'Tanker', 'Tug']}

    
    #ResNet models to use for each dataset
    student_model = args.student_model
    teacher_model = args.teacher_model
    ablation = args.ablation
    max_level = args.max_level
    
    # if ablation:
    #     folder_name = 'task_flag'
    
    patience = args.patience
    temperature = args.temperature
    mode = args.mode
    model_group = args.model_group

    
    #Number of classes in each dataset
    num_classes = {'DeepShip': 4}
    
    #Number of runs and/or splits for each dataset
    Splits = {'DeepShip':3}
    
    #Number of runs and/or splits for each dataset
    TDNN_feats = {'DeepShip': 1}
    TDDN_feats_teacher ={'DeepShip': 3}
    window_length = {'DeepShip': 256
        }
    hop_length = {'DeepShip': 96}
    
    Dataset = Dataset_names[data_selection]
    data_dir = Data_dirs[Dataset]
    
    #Save results based on features
    if (Dataset=='DeepShip'):
        audio_features = True
    else:
        audio_features = False
    

    Hist_model_name = 'Hist{}_{}'.format(student_model,numBins)
    Hist_model_name_teacher = 'Hist{}_{}'.format(teacher_model,numBins)
    
    #Return dictionary of parameters
    Params = {'save_results': save_results,'folder': folder,
                          'histogram': histogram,'Dataset': Dataset, 'data_dir': data_dir,'segment_length':segment_length,
                          'optimizer': optimizer,'HPRC':HPRC,
                          'num_workers': num_workers, 'method': method,'lr': lr,
                          'step_size': step_size,'class_names':class_names,'task_flag':task_flag,'ablation':ablation,
                          'gamma': gamma, 'batch_size' : batch_size, 
                          'num_epochs': num_epochs, 'model':model,
                          'padding': padding, 'max_level':max_level,
                          'stride': stride, 'kernel_size': kernel_size,
                          'in_channels': in_channels,'out_channels': out_channels,
                          'normalize_count': normalize_count, 
                          'normalize_bins': normalize_bins,'parallel': parallel,
                          'numBins': numBins,'feat_map_size': feat_map_size,
                          'student_model': student_model, 'teacher_model':teacher_model,'num_classes': num_classes, 
                          'Splits': Splits, 'feature_extraction': feature_extraction,'use_pretrained': use_pretrained,
                          'hist_model': Hist_model_name, 'hist_model_teacher':Hist_model_name_teacher,'use_pretrained': use_pretrained,
                          'add_bn': add_bn, 'pin_memory': pin_memory, 'scale': scale,
                          'TSNE_visual': TSNE_visual,
                          'Parallelize': Parallelize_model,
                          'Num_TSNE_images': Num_TSNE_images,'fig_size': fig_size,'level_num':level_num,
                          'font_size': font_size, 'feature': feature, 
                          'TDNN_feats': TDNN_feats,'TDDN_feats_teacher':TDDN_feats_teacher, 'window_length': window_length, 
                          'hop_length': hop_length,'audio_features': audio_features,
                          'sample_rate': sample_rate,'patience':patience,'temperature':temperature,'mode':mode,'model_group':model_group}
    
    return Params