import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, metrics
from tensorflow.keras.models import Model, load_model
import tensorflow.keras.backend as K

class DenseNet:

    def __init__(self,
                 input_shape,
                 num_class,
                 init_filters,
                 depth,
                 pocket):

        # User defined inputs
        self.input_shape = input_shape
        self.init_filters = init_filters
        self.num_class = num_class
        self.depth = depth
        self.pocket = pocket

        # Two convolution layers per block. I can play with this later to see if 3 or 4 improves
        # performance
        self.convs_per_block = 2

        # If pocket network, do not double feature maps after downsampling
        self.mul_on_downsample = 2
        if self.pocket:
            self.mul_on_downsample = 1

        # Parameters for each keras layer that we use.
        # I like to keep them all in one place (i.e., a params dictionary)
        self.params = dict()
        self.params['conv'] = dict(kernel_size = 3, padding = 'same')
        self.params['point_conv'] = dict(kernel_size = 1, padding = 'same')
        self.params['trans_conv'] = dict(kernel_size = (2, 2, 2), strides = (2, 2, 2), padding = 'same')

    def conv(self, x, filters):
        x = layers.Conv3D(filters = filters, **self.params['conv'])(x)
        x = tfa.layers.InstanceNormalization(axis = -1,
                                             center = True,
                                             scale = True,
                                             beta_initializer = 'random_uniform',
                                             gamma_initializer = 'random_uniform')(x)
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def block(self, x, filters):
        for _ in range(self.convs_per_block):
            y = self.conv(x, filters)
            x = layers.concatenate([x, y])

        x = layers.Conv3D(filters, kernel_size = 1)(x)
        x = layers.PReLU(shared_axes=[1, 2, 3])(x)
        return x

    def squeeze_excite_block(self, x, filters):

        '''
        Squeeze-Excitation Block from:
        https://arxiv.org/pdf/1709.01507.pdf

        Code adapted from:
        https://github.com/titu1994/keras-squeeze-excite-network
        '''

        se_shape = (1, 1, filters)
        ratio = 2
        se = layers.GlobalAveragePooling3D()(x)
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters // ratio, activation = 'relu', use_bias = False)(se)
        se = layers.Dense(filters, activation = 'sigmoid', use_bias = False)(se)

        x = layers.Add()([layers.multiply([x, se]), x])
        return x

    def encoder(self, x):
        skips = list()
        for i in range(self.depth):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            skips.append(self.squeeze_excite_block(self.block(x, filters), filters))
            x = layers.Conv3D(filters, kernel_size = (2, 2, 2), strides = (2, 2, 2))(skips[i])
            x = layers.PReLU(shared_axes=[1, 2, 3])(x)

        # Bottleneck
        filters = self.init_filters * (self.mul_on_downsample) ** self.depth
        skips.append(self.squeeze_excite_block(self.block(x, filters), filters))
        return skips

    def decoder(self, skips):
        x = skips[-1]
        skips = skips[:-1]
        deepsuper_blocks = list()
        for i in range(self.depth - 1, -1, -1):
            filters = self.init_filters * (self.mul_on_downsample) ** i
            x = layers.Conv3DTranspose(filters, **self.params['trans_conv'])(x)
            x = layers.PReLU(shared_axes=[1, 2, 3])(x)

            x = layers.concatenate([x, skips[i]])
            x = self.block(x, filters)
            x = self.squeeze_excite_block(x, filters)
            deepsuper_blocks.append(x)

        # Use deep supervision
        d = layers.Conv3D(self.num_class, kernel_size = 1)(deepsuper_blocks[0])
        d = layers.UpSampling3D(size = (2, 2, 2))(d)
        for i in range(1, len(deepsuper_blocks) - 1):
            d_next = layers.Conv3D(self.num_class, kernel_size = 1)(deepsuper_blocks[i])
            d = layers.Add()([d, d_next])
            d = layers.UpSampling3D(size = (2, 2, 2))(d)

        x = layers.Conv3D(self.num_class, 1)(x)
        x = layers.Add()([x, d])
        x = layers.Activation('softmax', dtype = 'float32')(x)

        return x


    def build_model(self):
        inputs = layers.Input(self.input_shape)

        skips = self.encoder(inputs)
        outputs = self.decoder(skips)

        model = Model(inputs = [inputs], outputs = [outputs])

        return model
