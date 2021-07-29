import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import os

import pdb

class Dataset:
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)
            
        self.df = pd.read_csv(self.params['raw_paths_csv'])
        self.n_channels = len(self.params['images'])
        self.n_classes = len(self.params['labels'])
        
    def decode(self, serialized_example):
        # Create features dictionary 
        features_dict = {'image': tf.io.FixedLenFeature([*self.params['patch_size'], self.n_channels], tf.float32),
                         'mask': tf.io.FixedLenFeature([*self.params['patch_size'], self.n_classes], tf.float32)}

        # Decode examples stored in TFRecord
        features = tf.io.parse_single_example(serialized_example, features_dict)

        return features['image'], features['mask']
    
    def configure_ds_for_performance(self, ds):
        ds = ds.shuffle(buffer_size = 50, reshuffle_each_iteration = True)
        ds = ds.batch(batch_size = self.params['batch_size'], drop_remainder = True)
        ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    def train_val_split(self):
        
        train_df, val_df, _, _ = train_test_split(self.df, 
                                                  self.df, 
                                                  test_size = 0.2, 
                                                  random_state = 42)
        
        train_df = train_df.reset_index(drop = True)
        val_df = val_df.reset_index(drop = True)
        
        train_tfr_list = list()
        for i in range(len(train_df)):
            patient_id = train_df.iloc[i]['id']
            filename = os.path.join(self.params['processed_data_dir'], '{}.tfrecord'.format(patient_id))
            train_tfr_list.append(filename)
        
        val_tfr_list = list()
        for i in range(len(val_df)):
            patient_id = val_df.iloc[i]['id']
            filename = os.path.join(self.params['processed_data_dir'], '{}.tfrecord'.format(patient_id))
            val_tfr_list.append(filename)
        
        train_ds = tf.data.TFRecordDataset(train_tfr_list, 
                                           compression_type = 'GZIP', 
                                           num_parallel_reads = tf.data.AUTOTUNE)
        train_ds = train_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE)
        train_ds = self.configure_ds_for_performance(train_ds)
        
        val_ds = tf.data.TFRecordDataset(val_tfr_list, 
                                         compression_type = 'GZIP', 
                                         num_parallel_reads = tf.data.AUTOTUNE)
        val_ds = val_ds.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE)
        val_ds = self.configure_ds_for_performance(val_ds)
        
        return train_ds, val_ds, val_df
    
    def crossval_splits(self):        
        train_splits = list()
        val_splits = list()
        val_df_splits = list()
        
        kfold = KFold(n_splits = 5, shuffle = True, random_state = 42)
        tfrecords = [os.path.join(self.params['processed_data_dir'], 
                                  '{}.tfrecord'.format(self.df.iloc[i]['id'])) for i in range(len(self.df))]
        splits = kfold.split(tfrecords)

        # For each split, train a model and predict on validation data. Write validation predictions as .nii.gz files.
        for split in splits:
            train = [tfrecords[idx] for idx in split[0]]
            val = [tfrecords[idx] for idx in split[1]]
            val_df_ids = [self.df.iloc[idx]['id'] for idx in split[1]]
            val_df = self.df.loc[self.df['id'].isin(val_df_ids)].reset_index(drop = True)
            val_df_splits.append(val_df)
            
            train = tf.data.TFRecordDataset(list(train), 
                                            compression_type = 'GZIP', 
                                            num_parallel_reads = tf.data.AUTOTUNE)
            train = train.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE)
            train = self.configure_ds_for_performance(train)
            train_splits.append(train)

            val = tf.data.TFRecordDataset(list(val), 
                                          compression_type = 'GZIP', 
                                          num_parallel_reads = tf.data.AUTOTUNE)
            val = val.map(self.decode, num_parallel_calls = tf.data.AUTOTUNE)
            val = self.configure_ds_for_performance(val)
            val_splits.append(val)
            
        return train_splits, val_splits, val_df_splits