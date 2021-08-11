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

    def preprocess_inference(self, patient_dict, mode):
        image_list = list(patient_dict.values())[2:len(patient_dict)]
        dims = ants.image_read(image_list[0]).numpy().shape
        if mode == 'test':
            patch_radius = [patch_dim // 2 for patch_dim in self.params['patch_size']]
        elif mode == 'train':
            patch_radius = self.params['patch_size']
        pad_dims = list()
        negative_pad_width = list()
        for i in range(3):
            if dims[i] % patch_radius[i] == 0:
                pad_dims.append(int(dims[i]))
                negative_pad_width.append(0)
            else:
                negative_pad_width.append(-1 * (int(np.ceil(dims[i] / patch_radius[i]) * patch_radius[i]) - dims[i]))
                pad_dims.append(int(np.ceil(dims[i] / patch_radius[i]) * patch_radius[i]))

        # Apply windowing and normalization to images
        image_npy = np.zeros((*pad_dims, len(image_list)))
        for i in range(len(image_list)):
            img = ants.image_read(image_list[i])
            img = ants.pad_image(img, shape = pad_dims)
            img = img.numpy()
            img = self.window(img)
            img = self.normalize(img)
            image_npy[..., i] = img

        return image_npy, dims, pad_dims, negative_pad_width

    def float_feature(self, value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = value))

    def int_feature(self, value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

    def run(self):
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            patient_tfr_name = '{}.tfrecord'.format(patient['id'])
            filename = os.path.join(self.params['processed_data_dir'], patient_tfr_name)
            writer = tf.io.TFRecordWriter(filename,
                                          options = tf.io.TFRecordOptions(compression_type = 'GZIP'))

            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[2:len(patient)]
            mask = ants.image_read(patient['mask'])

            # Create brainmask from first image in image list
            nzmask = ants.image_read(image_list[0])
            nzmask = ants.get_mask(nzmask, cleanup = 0)

            # Crop mask and all images according to brainmask and pad with patch radius
            mask_crop = ants.crop_image(mask, nzmask)
            mask_crop = ants.pad_image(mask_crop, pad_width = self.params['patch_size'])

            images_crop = list()
            for image in image_list:
                cropped = ants.image_read(image)
                cropped = ants.crop_image(cropped, nzmask)
                cropped = ants.pad_image(cropped, pad_width = self.params['patch_size'])
                images_crop.append(cropped)

            # Get dims of cropped images
            mask_npy = mask_crop.numpy()
            dims = mask_npy.shape

            # One hot encode mask and apply padding
            mask_onehot = np.empty((*dims, len(2 * self.params['labels'])))
            for j in range(len(self.params['labels'])):
                mask_onehot[..., j] = mask_npy == self.params['labels'][j]
                mask_onehot[..., j + len(self.params['labels'])] = self.get_dtm(mask_onehot[..., j])

            # Apply windowing and normalization to images
            image_npy = np.zeros((*dims, len(image_list)))
            for j in range(len(image_list)):
                img = images_crop[j].numpy()
                img = self.window(img)
                img = self.normalize(img)
                image_npy[..., j] = img

            # Get points in mask associated with each label
            label_points_list = list()
            num_label_points_list = [0]
            for j in range(len(self.params['labels'])):
                if self.params['labels'][j] == 0:
                    fg_mask = (mask_onehot[..., 0] == 0).astype('int')
                    label_mask = (image_npy[..., 0] != 0).astype('int') - fg_mask
                    nonzeros = np.nonzero(label_mask)
                    nonzeros = np.vstack((nonzeros[0], nonzeros[1], nonzeros[2]))
                    num_nonzeros = nonzeros.shape[-1]

                    label_points_list.append(nonzeros)
                    num_label_points_list.append(num_nonzeros)
                else:
                    label_mask = mask_onehot[..., j]
                    nonzeros = np.nonzero(label_mask)
                    nonzeros = np.vstack((nonzeros[0], nonzeros[1], nonzeros[2]))
                    num_nonzeros = nonzeros.shape[-1]

                    label_points_list.append(nonzeros)
                    num_label_points_list.append(num_nonzeros)

            # A 3xn matrix where each column is a point
            label_points = np.concatenate([label_points_list[i] for i in range(len(self.params['labels']))], axis = -1)

            # A list that stores the index ranges of each label in the label_points list
            # label_ranges = [0, # of label 0 points, # label 1 points, # label 2 points, ...]
            # We can sample a point in label one by picking a number x between [# label 0 points, # label 1 points]
            # and saying label_points[:, x]
            label_index_ranges = np.cumsum(num_label_points_list).astype('int')

            feature = {'image': self.float_feature(image_npy.ravel()),
                       'mask': self.float_feature(mask_onehot.ravel()),
                       'dims': self.int_feature(list(dims)),
                       'num_channels': self.int_feature([len(image_list)]),
                       'num_classes': self.int_feature([len(self.params['labels'])]),
                       'label_points': self.int_feature(label_points.ravel()),
                       'label_index_ranges': self.int_feature(list(label_index_ranges))}

            example = tf.train.Example(features = tf.train.Features(feature = feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())
            writer.close()
