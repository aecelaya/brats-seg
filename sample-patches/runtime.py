import pandas as pd
import numpy as np
import ants
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import os
import json
from tqdm import trange

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics, mixed_precision
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import tensorflow.keras.backend as K

from model import *
from loss import *
from datagenerator import Dataset
from preprocess import Preprocess

import pdb

# Set this environment variable to allow ModelCheckpoint to work
os.environ['HDF5_USE_FILE_LOCKING'] = 'FALSE'

# Set mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# For tensorflow 2.x.x allow memory growth on GPU
###################################
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
###################################

class RunTime:
    
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)

        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
        self.infer_preprocessor = Preprocess(json_file)
        self.dataset_builder = Dataset(json_file)
        
    def get_factors(self, n):    
            return [i for i in range(1, n + 1) if n % i == 0]
        
    def get_strides(self, image_dims):
        strides = list()
        for i in range(len(image_dims)):
            factors = self.get_factors(image_dims[i] - self.params['patch_size'][i])
            factors.sort()
            strides.append(np.max([factor for factor in factors if factor < self.params['patch_size'][i]]))
        return strides
        
    def val_inference(self, model, df):
        for i in trange(len(df)):
            patient = df.iloc[i].to_dict()
            image, original_dims, pad_dims, pad_width = self.infer_preprocessor.preprocess_inference(patient)  
                
            strides = [patch_dim // 2 for patch_dim in self.params['patch_size']]
            prediction = np.zeros((*pad_dims, len(self.params['labels'])))
            for i in range(0, pad_dims[0] - self.params['patch_size'][0] + 1, strides[0]):
                for j in range(0, pad_dims[1] - self.params['patch_size'][1] + 1, strides[1]):
                    for k in range(0, pad_dims[2] - self.params['patch_size'][2] + 1, strides[2]):
                        patch = image[i:(i + self.params['patch_size'][0]), 
                                      j:(j + self.params['patch_size'][1]), 
                                      k:(k + self.params['patch_size'][2]), ...]
                        patch = patch.reshape((1, *patch.shape))
                        pred_patch = model.predict(patch)
                        prediction[i:(i + self.params['patch_size'][0]), 
                                   j:(j + self.params['patch_size'][1]), 
                                   k:(k + self.params['patch_size'][2]), ...] = pred_patch
        
            prediction = np.argmax(prediction, axis = -1).astype('float32')
            prediction = ants.from_numpy(data = prediction)
            prediction = ants.pad_image(prediction, shape = [original_dims])

            # Convert prediction from numpy to ants image
            mask = ants.image_read(patient['mask'])
            prediction = ants.copy_image_info(mask, prediction)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction, 
                             os.path.join(self.params['prediction_dir'], prediction_filename))
            
    def trainval(self):
        train, val, val_df = self.dataset_builder.train_val_split()
        
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = DenseNet(input_shape = (*self.params['patch_size'], self.n_channels), 
                             num_class = self.n_classes, 
                             init_filters = 16, 
                             depth = 4, 
                             pocket = True).build_model()
            
        #opt = tfa.optimizers.NovoGrad()
        model.compile(optimizer = 'adam', loss = [dice_loss_weighted])
        
        # Reduce learning rate by 0.5 if validation dice coefficient does not improve after 5 epochs
        reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                                      mode = 'min',
                                      factor = 0.5, 
                                      patience = 5, 
                                      min_lr = 0.000001, 
                                      verbose = 1)

        model_name = os.path.join(self.params['model_dir'], 'brats_model_train_val_split')
        save_best = ModelCheckpoint(filepath = model_name, 
                                    monitor = 'val_loss', 
                                    verbose = 1, 
                                    save_best_only = True)

        # Train model
        model.fit(train, 
                  epochs = self.params['epochs'], 
                  steps_per_epoch = 500, 
                  validation_data = val, 
                  validation_steps = 250, 
                  callbacks = [reduce_lr, save_best], 
                  verbose = 1)
        
        model = load_model(model_name, custom_objects = {'dice_loss_weighted': dice_loss_weighted})
        self.val_inference(model, val_df)
        
    def crossval(self):
        train, val, val_df = self.dataset_builder.crossval_splits()
        
        for i in range(len(train)):
            print('Starting split {} of {}'.format(i + 1, len(train)))
            mirrored_strategy = tf.distribute.MirroredStrategy()
            with mirrored_strategy.scope():
                model = DenseNet(input_shape = (*self.params['patch_size'], self.n_channels), 
                                 num_class = self.n_classes, 
                                 init_filters = 16, 
                                 depth = 4, 
                                 pocket = True).build_model()

            model.compile(optimizer = 'adam', loss = [dice_loss_weighted])
            
            # Reduce learning rate by 0.5 if validation dice coefficient does not improve after 5 epochs
            reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', 
                                          mode = 'min',
                                          factor = 0.5, 
                                          patience = 5, 
                                          min_lr = 0.000001, 
                                          verbose = 1)
            
            model_name = os.path.join(self.params['model_dir'], 'brats_model_split_{}'.format(i + 1))
            save_best = ModelCheckpoint(filepath = model_name, 
                        monitor = 'val_loss', 
                        verbose = 1, 
                        save_best_only = True)
            
            # Train model
            model.fit(train[i], 
                      epochs = self.params['epochs'], 
                      steps_per_epoch = 500, 
                      validation_data = val[i], 
                      validation_steps = 250, 
                      callbacks = [reduce_lr, save_best], 
                      verbose = 1)
            
            model = load_model(model_name, custom_objects = {'dice_loss_weighted': dice_loss_weighted})
            self.val_inference(model, val_df[i])
            
    def run(self):
        if self.params['train_proto'] == 'trainval':
            self.trainval()
        else:
            self.crossval()
