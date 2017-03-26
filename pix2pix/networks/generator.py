from keras.layers import Activation, Input, Dropout, merge
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model

"""
There are two models available for the generator:
1. AE Generator
2. UNet with skip connections
"""


def make_generator_ae(input_layer, num_output_filters):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    :param model:
    :return:
    """
    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    filter_sizes = [64, 128, 256, 512, 512, 512, 512, 512]

    encoder = input_layer
    for filter_size in filter_sizes:
        encoder = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(encoder)
        # paper skips batch norm for first layer
        if filter_size != 64:
            encoder = BatchNormalization()(encoder)
        encoder = Activation(LeakyReLU(alpha=0.2))(encoder)

    # -------------------------------
    # DECODER
    # CD512-CD512-CD512-C512-C512-C256-C128-C64
    # 1 layer block = Conv - Upsample - BN - DO - Relu
    # -------------------------------
    stride = 2
    filter_sizes = [512, 512, 512, 512, 512, 256, 128, 64]

    decoder = encoder
    for filter_size in filter_sizes:
        decoder = UpSampling2D(size=(2, 2))(decoder)
        decoder = Convolution2D(nb_filter=filter_size, nb_row=4, nb_col=4, border_mode='same')(decoder)
        decoder = BatchNormalization()(decoder)
        decoder = Dropout(p=0.5)(decoder)
        decoder = Activation('relu')(decoder)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    decoder = Convolution2D(nb_filter=num_output_filters, nb_row=4, nb_col=4, border_mode='same')(decoder)
    generator = Activation('tanh')(decoder)
    return generator


def UNETGenerator(input_img_dim, num_output_channels):
    """
    Creates the generator according to the specs in the paper below.
    It's basically a skip layer AutoEncoder

    Generator does the following:
    1. Takes in an image
    2. Generates an image from this image

    Differs from a standard GAN because the image isn't random.
    This model tries to learn a mapping from a suboptimal image to an optimal image.

    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]
    :param input_img_dim: (channel, height, width)
    :param output_img_dim: (channel, height, width)
    :return:
    """
    # -------------------------------
    # ENCODER
    # C64-C128-C256-C512-C512-C512-C512-C512
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    merge_mode = 'concat'

    # batch norm mode
    bn_mode = 2

    # batch norm merge axis
    bn_axis = 1

    input_layer = Input(shape=input_img_dim, name="unet_input")

    # 1 encoder C64
    # skip batchnorm on this layer on purpose (from paper)
    en_1 = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(input_layer)
    en_1 = LeakyReLU(alpha=0.2)(en_1)

    # 2 encoder C128
    en_2 = Convolution2D(nb_filter=128, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_1)
    en_2 = BatchNormalization(name='gen_en_bn_2', mode=bn_mode, axis=bn_axis)(en_2)
    en_2 = LeakyReLU(alpha=0.2)(en_2)

    # 3 encoder C256
    en_3 = Convolution2D(nb_filter=256, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_2)
    en_3 = BatchNormalization(name='gen_en_bn_3', mode=bn_mode, axis=bn_axis)(en_3)
    en_3 = LeakyReLU(alpha=0.2)(en_3)

    # 4 encoder C512
    en_4 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_3)
    en_4 = BatchNormalization(name='gen_en_bn_4', mode=bn_mode, axis=bn_axis)(en_4)
    en_4 = LeakyReLU(alpha=0.2)(en_4)

    # 5 encoder C512
    en_5 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_4)
    en_5 = BatchNormalization(name='gen_en_bn_5', mode=bn_mode, axis=bn_axis)(en_5)
    en_5 = LeakyReLU(alpha=0.2)(en_5)

    # 6 encoder C512
    en_6 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_5)
    en_6 = BatchNormalization(name='gen_en_bn_6', mode=bn_mode, axis=bn_axis)(en_6)
    en_6 = LeakyReLU(alpha=0.2)(en_6)

    # 7 encoder C512
    en_7 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_6)
    en_7 = BatchNormalization(name='gen_en_bn_7', mode=bn_mode, axis=bn_axis)(en_7)
    en_7 = LeakyReLU(alpha=0.2)(en_7)

    # 8 encoder C512
    en_8 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride))(en_7)
    en_8 = BatchNormalization(name='gen_en_bn_8', mode=bn_mode, axis=bn_axis)(en_8)
    en_8 = LeakyReLU(alpha=0.2)(en_8)

    # -------------------------------
    # DECODER
    # CD512-CD1024-CD1024-C1024-C1024-C512-C256-C128
    # 1 layer block = Conv - Upsample - BN - DO - Relu
    # also adds skip connections (merge). Takes input from previous layer matching encoder layer
    # -------------------------------
    # 1 decoder CD512 (decodes en_8)
    de_1 = UpSampling2D(size=(2, 2))(en_8)
    de_1 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same')(de_1)
    de_1 = BatchNormalization(name='gen_de_bn_1', mode=bn_mode, axis=bn_axis)(de_1)
    de_1 = Dropout(p=0.5)(de_1)
    de_1 = merge([de_1, en_7], mode=merge_mode, concat_axis=1)
    de_1 = Activation('relu')(de_1)

    # 2 decoder CD1024 (decodes en_7)
    de_2 = UpSampling2D(size=(2, 2))(de_1)
    de_2 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_2)
    de_2 = BatchNormalization(name='gen_de_bn_2', mode=bn_mode, axis=bn_axis)(de_2)
    de_2 = Dropout(p=0.5)(de_2)
    de_2 = merge([de_2, en_6], mode=merge_mode, concat_axis=1)
    de_2 = Activation('relu')(de_2)

    # 3 decoder CD1024 (decodes en_6)
    de_3 = UpSampling2D(size=(2, 2))(de_2)
    de_3 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_3)
    de_3 = BatchNormalization(name='gen_de_bn_3', mode=bn_mode, axis=bn_axis)(de_3)
    de_3 = Dropout(p=0.5)(de_3)
    de_3 = merge([de_3, en_5], mode=merge_mode, concat_axis=1)
    de_3 = Activation('relu')(de_3)

    # 4 decoder CD1024 (decodes en_5)
    de_4 = UpSampling2D(size=(2, 2))(de_3)
    de_4 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_4)
    de_4 = BatchNormalization(name='gen_de_bn_4', mode=bn_mode, axis=bn_axis)(de_4)
    de_4 = Dropout(p=0.5)(de_4)
    de_4 = merge([de_4, en_4], mode=merge_mode, concat_axis=1)
    de_4 = Activation('relu')(de_4)

    # 5 decoder CD1024 (decodes en_4)
    de_5 = UpSampling2D(size=(2, 2))(de_4)
    de_5 = Convolution2D(nb_filter=1024, nb_row=4, nb_col=4, border_mode='same')(de_5)
    de_5 = BatchNormalization(name='gen_de_bn_5', mode=bn_mode, axis=bn_axis)(de_5)
    de_5 = Dropout(p=0.5)(de_5)
    de_5 = merge([de_5, en_3], mode=merge_mode, concat_axis=1)
    de_5 = Activation('relu')(de_5)

    # 6 decoder C512 (decodes en_3)
    de_6 = UpSampling2D(size=(2, 2))(de_5)
    de_6 = Convolution2D(nb_filter=512, nb_row=4, nb_col=4, border_mode='same')(de_6)
    de_6 = BatchNormalization(name='gen_de_bn_6', mode=bn_mode, axis=bn_axis)(de_6)
    de_6 = Dropout(p=0.5)(de_6)
    de_6 = merge([de_6, en_2], mode=merge_mode, concat_axis=1)
    de_6 = Activation('relu')(de_6)

    # 7 decoder CD256 (decodes en_2)
    de_7 = UpSampling2D(size=(2, 2))(de_6)
    de_7 = Convolution2D(nb_filter=256, nb_row=4, nb_col=4, border_mode='same')(de_7)
    de_7 = BatchNormalization(name='gen_de_bn_7', mode=bn_mode, axis=bn_axis)(de_7)
    de_7 = Dropout(p=0.5)(de_7)
    de_7 = merge([de_7, en_1], mode=merge_mode, concat_axis=1)
    de_7 = Activation('relu')(de_7)

    # After the last layer in the decoder, a convolution is applied
    # to map to the number of output channels (3 in general,
    # except in colorization, where it is 2), followed by a Tanh
    # function.
    de_8 = UpSampling2D(size=(2, 2))(de_7)
    de_8 = Convolution2D(nb_filter=num_output_channels, nb_row=4, nb_col=4, border_mode='same')(de_8)
    de_8 = Activation('tanh')(de_8)

    unet_generator = Model(input=[input_layer], output=[de_8], name='unet_generator')
    return unet_generator