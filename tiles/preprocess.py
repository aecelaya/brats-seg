import pandas as pd
import numpy as np
import os
import ants
import json
from tqdm import trange
import tensorflow as tf

class Preprocess:
    
    def __init__(self, json_file):
        with open(json_file, 'r') as file:
            self.params = json.load(file)
            
        self.df = self.get_files_df()
        self.df.to_csv(self.params['raw_paths_csv'], index = False)
        self.window_range = [0.5, 99.5]
        self.tile_width = self.params['tile_dims'][-1]
            
    def get_files_list(self, path):
        files_list = list()
        for root, _, files in os.walk(path, topdown = False):
            for name in files:
                files_list.append(os.path.join(root, name))
        return files_list
            
    def get_files_df(self):
        base_dir = os.path.abspath(self.params['raw_data_dir'])
        names_dict = dict()
        names_dict['mask'] = self.params['mask']
        for key in self.params['images'].keys():
            names_dict[key] = self.params['images'][key]
            
        cols = ['id'] + list(names_dict.keys())
        df = pd.DataFrame(columns = cols)
        row_dict = dict.fromkeys(cols)

        ids = os.listdir(base_dir)

        for i in ids:
            row_dict['id'] = i
            path = os.path.join(base_dir, i)
            files = self.get_files_list(path)

            for file in files:
                for img_type in names_dict.keys():
                    for img_string in names_dict[img_type]:
                        if img_string in file:
                            row_dict[img_type] = file

            df = df.append(row_dict, ignore_index = True)
        return df
    
    def window(self, image):
        # Input is a numpy array
        mask = (image != 0).astype('float32')
        nonzeros = image[image != 0]
        lower = np.percentile(nonzeros, self.window_range[0])
        upper = np.percentile(nonzeros, self.window_range[1])
        image = np.clip(image, lower, upper)
        image = np.multiply(mask, image)
        return image
    
    def normalize(self, image):
        # Input is a numpy array
        mask = (image != 0).astype('float32')
        nonzeros = image[image != 0]
        mean = np.mean(nonzeros)
        std = np.std(nonzeros)
        image = (image - mean) / std
        image = np.multiply(mask, image)
        return image
    
    def preprocess_inference(self, patient_dict):
        image_list = list(patient_dict.values())[2:len(patient_dict)]
        dims = ants.image_read(patient_dict['mask']).numpy().shape
        
        # Apply windowing and normalization to images
        image_npy = np.empty((*dims, len(image_list)))
        for i in range(len(image_list)):
            img = ants.image_read(image_list[i]).numpy()
            img = self.window(img)
            img = self.normalize(img)
            image_npy[..., i] = img
            
        return image_npy
    
    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = value))
    
    def run(self):
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            patient_tfr_name = '{}.tfrecord'.format(patient['id'])
            filename = os.path.join(self.params['processed_data_dir'], patient_tfr_name)
            writer = tf.io.TFRecordWriter(filename, 
                                          options = tf.io.TFRecordOptions(compression_type = 'GZIP'))
            
            image_list = list(patient.values())[2:len(patient)]
            mask = ants.image_read(patient['mask'])
            mask_npy = mask.numpy()
            dims = mask_npy.shape

            # One hot encode mask and apply padding
            mask_labels = self.params['labels']
            mask_onehot = np.empty((*dims, len(mask_labels)))
            for j in range(len(mask_labels)):
                mask_onehot[..., j] = mask_npy == mask_labels[j]
                            
            # Apply windowing and normalization to images
            image_npy = np.empty((*dims, len(image_list)))
            for j in range(len(image_list)):
                img = ants.image_read(image_list[j]).numpy()
                img = self.window(img)
                img = self.normalize(img)
                image_npy[..., j] = img
                
            for j in range(0, dims[-1] - self.params['tile_dims'][-1] + 1, 3):
                mask_slice = mask_onehot[:, :, j:(j + self.params['tile_dims'][-1]), :]

                mask_fg_slice = (mask_slice[..., 0] == 0).astype('float32')
                # Only take slices with tumor in them - this is for training only
                if  np.sum(mask_fg_slice) > 1000:

                    # Get corresponding image and multiclass mask slices
                    img_slice = image_npy[:, :, j:(j + self.params['tile_dims'][-1]), :]
                    
                    feature = {'image': self.float_feature(img_slice.ravel()), 
                               'mask': self.float_feature(mask_slice.ravel())}
                    example = tf.train.Example(features = tf.train.Features(feature = feature))
                    writer.write(example.SerializeToString())
                
            writer.close()
                
            
