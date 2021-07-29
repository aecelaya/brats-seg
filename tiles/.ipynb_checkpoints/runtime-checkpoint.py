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

# Set this environment variable to only use the first available GPU
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
        
    def val_inference(self, model, df):
        num_pats = len(df)
        for i in trange(num_pats):
            # Read and prepare image for inference
            patient = df.iloc[i].to_dict()
            dims = ants.image_read(patient['mask']).numpy().shape
            pred_img_depth = dims[-1] + (2 * self.params['tile_dims'][-1])
            image = np.empty((dims[0], dims[1], pred_img_depth, self.n_channels))
            image[:, 
                  :, 
                  self.params['tile_dims'][-1]:(pred_img_depth - self.params['tile_dims'][-1]), 
                  :] = self.infer_preprocessor.preprocess_inference(patient)

            # Predict on overlaping tiles of test image
            prediction_probs = np.empty((dims[0], dims[1], pred_img_depth, self.n_classes))
            for k in range(pred_img_depth - self.params['tile_dims'][-1] + 1):
                temp = image[:, :, k:(k + self.params['tile_dims'][-1]), :]
                temp = temp.reshape((1, *temp.shape))
                temp = model.predict(temp)
                temp = temp.reshape((dims[0], dims[1], self.params['tile_dims'][-1], self.n_classes))
                prediction_probs[:, :, k:(k + self.params['tile_dims'][-1]), :] += temp

            # Take average prediction from overlap strategy
            prediction_probs /= self.params['tile_dims'][-1]
            prediction_probs = prediction_probs[:, 
                                                :, 
                                                self.params['tile_dims'][-1]:(pred_img_depth - self.params['tile_dims'][-1]), 
                                                :]
        
            # Use argmax to get predicted classes
            prediction = np.argmax(prediction_probs, axis = -1).astype('float32')
            
            # Convert prediction from numpy to ants image
            mask = ants.image_read(patient['mask'])
            prediction = mask.new_image_like(data = prediction)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction, 
                             os.path.join(self.params['prediction_dir'], prediction_filename))
            
    def trainval(self):
        train, val, val_df = self.dataset_builder.train_val_split()
        
        mirrored_strategy = tf.distribute.MirroredStrategy()
        with mirrored_strategy.scope():
            model = DenseNet(input_shape = (*self.params['tile_dims'], self.n_channels), 
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
                model = DenseNet(input_shape = (*self.params['tile_dims'], self.n_channels), 
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
            
            model_name = os.path.join(self.params['model_dir'], 'rc_model_split_{}'.format(i + 1))
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
        
#     def crossval(self):
#         npy_df = pd.read_csv(self.params['npy_paths_csv'])
#         raw_df = pd.read_csv(self.params['raw_paths_csv'])
        
#         patients = np.unique(raw_df['id'])
#         kfold = KFold(n_splits = 5, shuffle = True)
#         splits = kfold.split(patients)
        
#         split_id = 1
#         for split in splits:
            
#             print('Starting split {} of {}'.format(split_id, self.n_folds))
            
#             train_pats = list(patients[split[0]])
#             val_pats = list(patients[split[1]])

#             # Create DataFrames with only training and validation patients for this split
#             train_df = npy_df[npy_df['id'].isin(train_pats)]
#             val_df = npy_df[npy_df['id'].isin(val_pats)]
#             val_images = raw_df[raw_df['id'].isin(val_pats)]

#             num_train_tiles = len(train_df)
#             num_val_tiles = len(val_df)

#             # Create training and validation data generators
#             train = DataGenerator(train_df, self.batch_size, (240, 240, self.params['tile_width']), self.n_channels, self.n_classes, True)
#             val = DataGenerator(val_df, self.batch_size, (240, 240, self.params['tile_width']), self.n_channels, self.n_classes, True)

#             model = DenseNet(input_shape = (*self.params['tile_dims'], self.n_channels), 
#                              num_class = self.n_classes, 
#                              init_filters = 16, 
#                              depth = 4, 
#                              pocket = True).build_model()
            
# #             cosine_warmup_lr = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate = 0.01, 
# #                                                                                  first_decay_steps = 1000, 
# #                                                                                  t_mul = 2.0, 
# #                                                                                  m_mul = 1.0, 
# #                                                                                  alpha = 0.0)

# #             opt = tf.keras.optimizers.SGD(learning_rate = cosine_lr, 
# #                                           momentum = 0.99, 
# #                                           nesterov = False)
#             model.compile(optimizer = 'adam', loss = [dice_loss_l2])
            
#             model_name = os.path.join(self.params['model_dir'], 'rc_model_split_{}'.format(split_id))
#             best_model = ModelCheckpoint(filepath = model_name, 
#                                          monitor = 'val_loss', 
#                                          verbose = 1, 
#                                          save_best_only = True)
            
#             model.fit(train, 
#                       epochs = self.params['epochs'], 
#                       steps_per_epoch = num_train_tiles // self.params['batch_size'], 
#                       validation_data = val, 
#                       validation_steps = num_val_tiles // self.params['batch_size'], 
#                       callbacks = [best_model], 
#                       verbose = 1)
            
#             model = load_model(model_name, custom_objects = {'dice_loss_l2': dice_loss_l2})
#             self.val_inference(model, val_images, self.params['prediction_dir'])
#             split_id += 1          

