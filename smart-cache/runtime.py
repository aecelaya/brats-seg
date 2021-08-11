import os
import json
import ants
import pandas as pd
import numpy as np
from tqdm import trange
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K

from model import *
from loss import *
from preprocess import *

import pdb

class RunTime:

    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)

        self.df = pd.read_csv(self.params['raw_paths_csv'])
        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
        self.n_folds = 5
        self.infer_preprocessor = Preprocess(json_file)

    def decode(self, serialized_example):
        features_dict = {'image': tf.io.VarLenFeature(tf.float32),
                         'mask': tf.io.VarLenFeature(tf.float32),
                         'dims': tf.io.FixedLenFeature([3], tf.int64),
                         'num_channels': tf.io.FixedLenFeature([1], tf.int64),
                         'num_classes': tf.io.FixedLenFeature([1], tf.int64),
                         'label_points': tf.io.VarLenFeature(tf.int64),
                         'label_index_ranges': tf.io.FixedLenFeature([len(self.params['labels']) + 1], tf.int64)}

        # Decode examples stored in TFRecord
        features = tf.io.parse_example(serialized_example, features_dict)

        # Crop random patch from images
        # Extract image/mask pair from sparse tensors
        image = tf.sparse.to_dense(features['image'])
        image = tf.reshape(image, tf.concat([features['dims'], features['num_channels']], axis = -1))

        mask = tf.sparse.to_dense(features['mask'])
        mask = tf.reshape(mask, tf.concat([features['dims'], 2 * features['num_classes']], axis = -1))

        # Extract point lists for each label
        # TF constant for reshaping list of points to array
        three = tf.constant(3, shape = (1,), dtype = tf.int64)
        label_index_ranges = features['label_index_ranges']
        num_points = tf.reshape(label_index_ranges[-1], shape = (1,))
        label_points = tf.sparse.to_dense(features['label_points'])
        label_points = tf.reshape(label_points, tf.concat([three, num_points], axis = -1))

        return image, mask, label_points, label_index_ranges

    def random_crop(self, image, mask, label_points, label_index_ranges, fg_prob):

        if tf.random.uniform([]) <= fg_prob:
            # Pick a foreground point (i.e., any label that is not 0)
            # Randomly pick a foreground class
            label_idx = tf.random.uniform([],
                                          minval = 1,
                                          maxval = len(self.params['labels']),
                                          dtype = tf.int32)
            low = label_idx
            high = label_idx + 1

            if label_index_ranges[high] <= label_index_ranges[low]:
                low = 1
                high = -1
        else:
            low = 0
            high = 1

        point_idx = tf.random.uniform([],
                                      minval = label_index_ranges[low],
                                      maxval = label_index_ranges[high],
                                      dtype=tf.int64)
        point = label_points[..., point_idx]

        # Extract random patch from image/mask
        patch_radius = [patch_dim // 2 for patch_dim in self.params['patch_size']]
        image_patch = image[point[0] - patch_radius[0]:point[0] + patch_radius[0],
                            point[1] - patch_radius[1]:point[1] + patch_radius[1],
                            point[2] - patch_radius[2]:point[2] + patch_radius[2],
                            ...]
        mask_patch = mask[point[0] - patch_radius[0]:point[0] + patch_radius[0],
                          point[1] - patch_radius[1]:point[1] + patch_radius[1],
                          point[2] - patch_radius[2]:point[2] + patch_radius[2],
                          ...]

        if self.params['augment']:
            # Random flips
            if tf.random.uniform([]) <= 0.15:
                axis = np.random.randint(0, 3)
                if axis == 0:
                    image_patch = image_patch[::-1, :, :, ...]
                    mask_patch = mask_patch[::-1, :, :, ...]
                elif axis == 1:
                    image_patch = image_patch[:, ::-1, :, ...]
                    mask_patch = mask_patch[:, ::-1, :, ...]
                else:
                    image_patch = image_patch[:, :, ::-1, ...]
                    mask_patch = mask_patch[:, :, ::-1, ...]

            # Add Rician noise
            if tf.random.uniform([]) <= 0.15:
                variance = tf.random.uniform([], minval = 0.001, maxval = 0.05)
                image_patch = tf.math.sqrt(
                    tf.math.square((image_patch + tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) +
                    tf.math.square(tf.random.normal(shape = tf.shape(image_patch), stddev = variance))) * tf.math.sign(image_patch)

            # Apply Gaussian blur on a per channel basis
            if tf.random.uniform([]) <= 0.15:
                blur_level = np.random.uniform(0.25, 0.75)
                image_patch = tf.numpy_function(scipy.ndimage.gaussian_filter,[image_patch, blur_level], tf.float32)

        return image_patch, mask_patch

    def val_inference(self, model, df):
        for i in trange(len(df)):
            patient = df.iloc[i].to_dict()
            image, original_dims, pad_dims, neg_pad_width = self.infer_preprocessor.preprocess_inference(patient_dict = patient,
                                                                                                         mode = 'test')

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

            prediction = np.argmax(prediction, axis = -1)
            prediction[prediction == 3] = 4
            prediction = prediction.astype('float32')
            prediction = ants.from_numpy(data = prediction)
            prediction = ants.pad_image(prediction, pad_width = neg_pad_width).numpy()

            # Convert prediction from numpy to ants image
            mask = ants.image_read(patient['mask'])
            prediction = mask.new_image_like(data = prediction)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction,
                             os.path.join(self.params['prediction_dir'], prediction_filename))

    def val_inference_mrf(self, model, df):
        for i in trange(len(df)):
            patient = df.iloc[i].to_dict()
            image, original_dims, pad_dims, neg_pad_width = self.infer_preprocessor.preprocess_inference(patient_dict = patient,
                                                                                                         mode = 'test')

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
                        pred_patch = pred_patch.reshape((*self.params['patch_size'], len(self.params['labels'])))
                        prediction[i:(i + self.params['patch_size'][0]),
                                   j:(j + self.params['patch_size'][1]),
                                   k:(k + self.params['patch_size'][2]), ...] = pred_patch

            # Start MRF regularization
            image_list = list(patient.values())[2:len(patient)]
            ants_images = list()
            for i in range(len(image_list)):
                ants_images.append(ants.image_read(image_list[i]))

            ants_prob_images = list()
            for i in range(len(self.params['labels'])):
                temp = ants.from_numpy(data = prediction[..., i])
                temp = ants.pad_image(temp, pad_width = neg_pad_width).numpy()
                ants_prob_images.append(ants_images[0].new_image_like(data = temp))

            nzmask = ants.get_mask(ants_images[0], cleanup = 0)
            prediction = ants.prior_based_segmentation(ants_images,
                                                       ants_prob_images,
                                                       nzmask)
            prediction = prediction['segmentation'].numpy() - 1
            prediction[prediction < 0] = 0
            prediction[prediction == 3] = 4
            prediction = ants_images[0].new_image_like(data = prediction)

            # Take only foreground components with 1000 voxels
            prediction_binary = ants.get_mask(prediction, cleanup = 0)
            prediction_binary = ants.label_clusters(prediction_binary, 1000)
            prediction_binary = ants.get_mask(prediction_binary, cleanup = 0)
            prediction_binary = prediction_binary.numpy()
            prediction = np.multiply(prediction_binary, prediction.numpy())

            # Copy header of original image to prediction
            prediction = ants_images[0].new_image_like(data = prediction)

            # Write prediction mask to nifti file and save to disk
            prediction_filename = '{}.nii.gz'.format(patient['id'])
            ants.image_write(prediction,
                             os.path.join(self.params['prediction_dir'], prediction_filename))

    def compute_val_loss(self, model, df):
        val_loss = list()
        for i in trange(len(df)):
            patient = df.iloc[i].to_dict()
            image, original_dims, pad_dims, neg_pad_width = self.infer_preprocessor.preprocess_inference(patient_dict = patient,
                                                                                                         mode = 'train')

            truth = np.zeros((*original_dims, len(self.params['labels'])))
            mask_npy = ants.image_read(patient['mask']).numpy()
            for i in range(len(self.params['labels'])):
                truth[..., i] = mask_npy == self.params['labels'][i]
            truth = truth.reshape((1, *truth.shape))

            #strides = [patch_dim // 2 for patch_dim in self.params['patch_size']]
            strides = self.params['patch_size']
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

            crop_prediction = np.zeros((*original_dims, len(self.params['labels'])))
            for i in range(len(self.params['labels'])):
                temp = ants.from_numpy(data = prediction[..., i])
                crop_prediction[..., i] = ants.pad_image(temp, pad_width = neg_pad_width).numpy()

            prediction = crop_prediction
            prediction = prediction.reshape((1, *prediction.shape))
            val_loss.append(dice_loss_weighted(truth, prediction))
        return np.mean(val_loss)

    def trainval(self):
        train_df, val_df, _, _ = train_test_split(self.df,
                                                  self.df,
                                                  test_size = 0.2,
                                                  random_state = 42)

        train_df = train_df.reset_index(drop = True)
        val_df = val_df.reset_index(drop = True)

        train_patients = list(train_df['id'])
        train_tfr_list = [os.path.join(self.params['processed_data_dir'], '{}.tfrecord'.format(patient_id)) for patient_id in train_patients]
        random.shuffle(train_tfr_list)

        current_model_name = os.path.join(self.params['model_dir'], '{}_current_model_trainval'.format(self.params['base_model_name']))
        best_model_name = os.path.join(self.params['model_dir'], '{}_best_model_trainval'.format(self.params['base_model_name']))

        best_val_loss = np.Inf
        for i in range(self.params['epochs']):
            print('Epoch {}/{}'.format(i + 1, self.params['epochs']))

            fg_prob = 0.85
            crop_fn = lambda image, mask, label_points, label_index_ranges: self.random_crop(image,
                                                                                             mask,
                                                                                             label_points,
                                                                                             label_index_ranges,
                                                                                             fg_prob)

            if i == 0:
                # Initialize training cache and pool in first epoch
                train_cache = random.sample(train_tfr_list, self.params['cache_size'])
                random.shuffle(train_cache)
                cache_pool = list(set(train_tfr_list) - set(train_cache))
                random.shuffle(cache_pool)

                # Build model from scratch in first epoch
                model = DenseNet(input_shape = (*self.params['patch_size'], self.n_channels),
                                 num_class = self.n_classes,
                                 init_filters = 16,
                                 depth = 4,
                                 pocket = True).build_model()
            else:
                # Pick n_replacement new patients from pool and remove the same number from the current cache
                n_replacements = int(np.ceil(self.params['cache_size'] * self.params['cache_replacement_rate']))
                new_cache_patients = random.sample(cache_pool, n_replacements)
                back_to_pool_patients = random.sample(train_cache, n_replacements)

                # Update cache and pool for next epoch
                train_cache = list(set(train_cache) - set(back_to_pool_patients)) + new_cache_patients
                random.shuffle(train_cache)
                cache_pool = list(set(cache_pool) - set(new_cache_patients)) + back_to_pool_patients
                random.shuffle(cache_pool)

                # Reload model and resume training for later epochs
                model = load_model(current_model_name, custom_objects = {'dice_loss_weighted': dice_loss_weighted})

            train_ds = tf.data.TFRecordDataset(train_cache,
                                               compression_type = 'GZIP',
                                               num_parallel_reads = tf.data.AUTOTUNE)
            train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE).cache()
            train_ds = train_ds.map(crop_fn, num_parallel_calls = tf.data.AUTOTUNE)
            train_ds = train_ds.batch(batch_size = self.params['batch_size'], drop_remainder = True)
            train_ds = train_ds.repeat()
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

            opt = tf.keras.optimizers.Adam()
            model.compile(optimizer = opt, loss = [dice_loss_weighted])

            # Train model
            model.fit(train_ds,
                      epochs = 1,
                      steps_per_epoch = 250)

            # Save model for next epoch
            model.save(current_model_name)

            # Comput loss for validation patients
            val_loss = self.compute_val_loss(model, val_df)
            if val_loss < best_val_loss:
                print('Val loss IMPROVED from {} to {}'.format(best_val_loss, val_loss))
                best_val_loss = val_loss
                model.save(best_model_name)
            else:
                print('Val loss of {} DID NOT IMPROVE from {}'.format(val_loss, best_val_loss))

            gc.collect()

        # Run prediction on validation set and write results to .nii.gz format
        model = load_model(best_model_name, custom_objects = {'dice_loss_weighted': dice_loss_weighted})
        self.val_inference_mrf(model, val_df)
        gc.collect()

    def crossval(self):
        train_splits = list()
        val_splits = list()
        val_df_splits = list()

        kfold = KFold(n_splits = self.n_folds, shuffle = True, random_state = 42)
        tfrecords = [os.path.join(self.params['processed_data_dir'],
                                  '{}.tfrecord'.format(self.df.iloc[i]['id'])) for i in range(len(self.df))]
        splits = kfold.split(tfrecords)

        split_cnt = 1
        for split in splits:
            print('Starting split {}/{}'.format(split_cnt, self.n_folds))
            train_tfr_list = [tfrecords[idx] for idx in split[0]]
            val_df_ids = [self.df.iloc[idx]['id'] for idx in split[1]]
            val_df = self.df.loc[self.df['id'].isin(val_df_ids)].reset_index(drop = True)

            current_model_name = os.path.join(self.params['model_dir'], '{}_current_model_trainval'.format(self.params['base_model_name']))
            best_model_name = os.path.join(self.params['model_dir'], '{}_best_model_trainval'.format(self.params['base_model_name']))

            best_val_loss = np.Inf
            for i in range(self.params['epochs']):
                print('Epoch {}/{}'.format(i + 1, self.params['epochs']))

                fg_prob = 0.85 #self.cosine_decay_step(step = i, initial = 0.95, final = 0.75)
                crop_fn = lambda image, mask, label_points, label_index_ranges: self.random_crop(image,
                                                                                                 mask,
                                                                                                 label_points,
                                                                                                 label_index_ranges,
                                                                                                 fg_prob)

                if i == 0:
                    # Initialize training cache and pool in first epoch
                    train_cache = random.sample(train_tfr_list, self.params['cache_size'])
                    random.shuffle(train_cache)
                    cache_pool = list(set(train_tfr_list) - set(train_cache))
                    random.shuffle(cache_pool)

                    # Build model from scratch in first epoch
                    model = DenseNet(input_shape = (*self.params['patch_size'], self.n_channels),
                                     num_class = self.n_classes,
                                     init_filters = 16,
                                     depth = 4,
                                     pocket = True).build_model()
                else:
                    # Pick n_replacement new patients from pool and remove the same number from the current cache
                    n_replacements = int(np.ceil(self.params['cache_size'] * self.params['cache_replacement_rate']))
                    new_cache_patients = random.sample(cache_pool, n_replacements)
                    back_to_pool_patients = random.sample(train_cache, n_replacements)

                    # Update cache and pool for next epoch
                    train_cache = list(set(train_cache) - set(back_to_pool_patients)) + new_cache_patients
                    random.shuffle(train_cache)
                    cache_pool = list(set(cache_pool) - set(new_cache_patients)) + back_to_pool_patients
                    random.shuffle(cache_pool)

                    # Reload model and resume training for later epochs
                    model = load_model(current_model_name, custom_objects = {'dice_norm_surf_loss': dice_norm_surf_loss_wrapper(alpha)})

                train_ds = tf.data.TFRecordDataset(train_cache,
                                                   compression_type = 'GZIP',
                                                   num_parallel_reads = tf.data.AUTOTUNE)
                train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE).cache()
                train_ds = train_ds.map(crop_fn, num_parallel_calls = tf.data.AUTOTUNE)
                train_ds = train_ds.batch(batch_size = self.params['batch_size'], drop_remainder = True)
                train_ds = train_ds.repeat()
                train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

                alpha = self.cosine_decay_step(i, initial = 1.0, final = 0.0)
                print('alpha = {}'.format(alpha))

                opt = tf.keras.optimizers.Adam()
                model.compile(optimizer = opt, loss = [dice_norm_surf_loss_wrapper(alpha)])

                # Train model
                model.fit(train_ds,
                          epochs = 1,
                          steps_per_epoch = 250)

                # Save model for next epoch
                model.save(current_model_name)

                # Comput loss for validation patients
                val_loss = self.compute_val_loss(model, val_df)
                if val_loss < best_val_loss:
                    print('Val loss IMPROVED from {} to {}'.format(best_val_loss, val_loss))
                    best_val_loss = val_loss
                    model.save(best_model_name)
                else:
                    print('Val loss of {} DID NOT IMPROVE from {}'.format(val_loss, best_val_loss))

                gc.collect()

            # Run prediction on validation set and write results to .nii.gz format
            model = load_model(best_model_name, custom_objects = {'dice_norm_surf_loss': dice_norm_surf_loss_wrapper(alpha)})
            self.val_inference(model, val_df)
            gc.collect()

    def run(self):
        if self.params['train_proto'] == 'trainval':
            self.trainval()
        else:
            self.crossval()
