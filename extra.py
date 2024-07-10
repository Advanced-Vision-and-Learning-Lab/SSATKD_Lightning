#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 20:54:39 2024

@author: jarin.ritu
"""

        elif args.mode == 'distillation':
            # pdb.set_trace()
            
            #Fine tune teacher on dataset
            teacher_checkpoint_callback = ModelCheckpoint(filename = 'best_model_teacher',mode='max',
                                                  monitor='val_accuracy')
            model_ft = Lightning_Wrapper(model.teacher, Params['num_classes'][Dataset], 
                                                  log_dir = filename, label_names=Params['class_names'])
            
            #Train teacher
            print("Setting up teacher trainer...")
            trainer_teacher = Trainer(callbacks=[EarlyStopping(monitor='val_loss', patience=Params['patience']), teacher_checkpoint_callback,
                                          TQDMProgressBar(refresh_rate=10)], 
                              max_epochs= Params['num_epochs'], enable_checkpointing = Params['save_results'], 
                              default_root_dir = filename,
                              logger=logger) 
            
            
            print("Teacher trainer set up.")
            
            # Start fitting the model
            print('Training teacher model...')
            
            trainer_teacher.fit(model_ft, train_dataloaders = train_loader, 
                                val_dataloaders = val_loader)
            print('Training completed.')
            
            #Pass fine-tuned teacher to knowledge distillation model
            sub_dir = generate_filename(Params, split)
            checkpt_path = os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/best_model_teacher.ckpt')             
            best_teacher = Lightning_Wrapper.load_from_checkpoint(checkpt_path,
                                                                  hparams_file=os.path.join(sub_dir, 'lightning_logs/Training/checkpoints/hparams.yaml'),
                                                                  model=model.teacher,
                                                                  num_classes = num_classes, 
                                                                  strict=True)
            model.teacher = best_teacher.model
        
            # Remove feature extraction layers from PANN/TIMM
            model.remove_PANN_feature_extractor()
            model.remove_TIMM_feature_extractor()
            
            model_ft = Lightning_Wrapper_KD(model, num_classes=Params['num_classes'][Dataset],  
                                         log_dir = filename, label_names=Params['class_names'],
                                         Params=Params,criterion=SSTKAD_Loss())