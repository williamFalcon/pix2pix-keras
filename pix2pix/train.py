import numpy as np
import os

import keras.backend as K
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Activation, Flatten, Dense, Input, Reshape, Dropout, merge, Lambda
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.models import Model
from pix2pix.utils.facades_generator import facades_generator
import time
from keras.utils import generic_utils as keras_generic_utils

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
DATASET = 'facades_bw'

def inverse_normalization(X):
    return X * 255.0

def plot_generated_batch(X_full, X_sketch, generator_model, epoch_num, dataset_name, batch_num):

    # Generate images
    X_gen = generator_model.predict(X_sketch)

    X_sketch = inverse_normalization(X_sketch)
    X_full = inverse_normalization(X_full)
    X_gen = inverse_normalization(X_gen)

    # limit to 8 images as output
    Xs = X_sketch[:8]
    Xg = X_gen[:8]
    Xr = X_full[:8]

    # put |decoded, generated, original| images next to each other
    X = np.concatenate((Xs, Xg, Xr), axis=3)

    # make one giant block of images
    X = np.concatenate(X, axis=1)

    # save the giant n x 3 images
    plt.imsave('./pix2pix_out/progress_imgs/{}_epoch_{}_batch_{}.png'.format(dataset_name, epoch_num, batch_num), X[0], cmap='Greys_r')


def get_disc_batch(X_original_batch, X_decoded_batch, generator_model, batch_counter, patch_dim,
                   label_smoothing=False, label_flipping=0):

    # Create X_disc: alternatively only generated or real images
    if batch_counter % 2 == 0:
        # generate fake image

        # Produce an output
        X_disc = generator_model.predict(X_decoded_batch)

        # each image will produce a 1x2 vector for the results (aka is fake or not)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

        # sets all first entries to 1. AKA saying these are fake
        # these are fake iamges
        y_disc[:, 0] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    else:
        # generate real image
        X_disc = X_original_batch

        # each image will produce a 1x2 vector for the results (aka is fake or not)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        if label_smoothing:
            y_disc[:, 1] = np.random.uniform(low=0.9, high=1, size=y_disc.shape[0])
        else:
            # these are real images
            y_disc[:, 1] = 1

        if label_flipping > 0:
            p = np.random.binomial(1, label_flipping)
            if p > 0:
                y_disc[:, [0, 1]] = y_disc[:, [1, 0]]

    # Now extract patches form X_disc
    X_disc = extract_patches(images=X_disc, sub_patch_dim=patch_dim)

    return X_disc, y_disc


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], X1.shape[0], replace=False)
        x1 = X1[idx]
        x2 = X2[idx]
        yield x1, x2

def minb_disc(x):
    diffs = K.expand_dims(x, 3) - K.expand_dims(K.permute_dimensions(x, [1, 2, 0]), 0)
    abs_diffs = K.sum(K.abs(diffs), 2)
    x = K.sum(K.exp(-abs_diffs), 2)

    return x

def lambda_output(input_shape):
    return input_shape[:2]

def num_patches(output_img_dim=(3, 256, 256), sub_patch_dim=(64, 64)):
    """
    Creates non-overlaping patches to feed to the PATCH GAN
    (Section 2.2.2 in paper)
    The paper provides 3 options.
    Pixel GAN = 1x1 patches (aka each pixel)
    PatchGAN = nxn patches (non-overlaping blocks of the image)
    ImageGAN = im_size x im_size (full image)

    Ex: 4x4 image with patch_size of 2 means 4 non-overlaping patches

    :param output_img_dim:
    :param sub_patch_dim:
    :return:
    """
    # num of non-overlaping patches
    nb_non_overlaping_patches = (output_img_dim[1] / sub_patch_dim[0]) * (output_img_dim[2] / sub_patch_dim[1])

    # dimensions for the patch discriminator
    patch_disc_img_dim = (output_img_dim[0], sub_patch_dim[0], sub_patch_dim[1])

    return int(nb_non_overlaping_patches), patch_disc_img_dim


def extract_patches(images, sub_patch_dim):
    """
    Cuts images into k subpatches
    Each kth cut as the kth patches for all images
    ex: input 3 images [im1, im2, im3]
    output [[im_1_patch_1, im_2_patch_1], ... , [im_n-1_patch_k, im_n_patch_k]]

    :param images: array of Images (num_images, im_channels, im_height, im_width)
    :param sub_patch_dim: (height, width) ex: (30, 30) Subpatch dimensions
    :return:
    """
    im_height, im_width = images.shape[2:]
    patch_height, patch_width = sub_patch_dim

    # list out all xs  ex: 0, 29, 58, ...
    x_spots = range(0, im_width, patch_width)

    # list out all ys ex: 0, 29, 58
    y_spots = range(0, im_height, patch_height)
    all_patches = []

    for y in y_spots:
        for x in x_spots:
            # indexing here is cra
            # images[num_images, num_channels, width, height]
            # this says, cut a patch across all images at the same time with this width, height
            image_patches = images[:, :, y: y+patch_height, x: x+patch_width]
            all_patches.append(np.asarray(image_patches, dtype=np.float32))
    return all_patches


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


def PatchGanDiscriminator(output_img_dim, patch_dim, nb_patches):
    """
    Creates the generator according to the specs in the paper below.
    [https://arxiv.org/pdf/1611.07004v1.pdf][5. Appendix]

    PatchGAN only penalizes structure at the scale of patches. This
    discriminator tries to classify if each N x N patch in an
    image is real or fake. We run this discriminator convolutationally
    across the image, averaging all responses to provide
    the ultimate output of D.

    The discriminator has two parts. First part is the actual discriminator
    seconds part we make it a PatchGAN by running each image patch through the model
    and then we average the responses

    Discriminator does the following:
    1. Runs many pieces of the image through the network
    2. Calculates the cost for each patch
    3. Returns the avg of the costs as the output of the network

    :param patch_dim: (channels, width, height) T
    :param nb_patches:
    :return:
    """
    # -------------------------------
    # DISCRIMINATOR
    # C64-C128-C256-C512-C512-C512 (for 256x256)
    # otherwise, it scales from 64
    # 1 layer block = Conv - BN - LeakyRelu
    # -------------------------------
    stride = 2
    bn_mode = 2
    axis = 1
    input_layer = Input(shape=patch_dim)

    # We have to build the discriminator dinamically because
    # the size of the disc patches is dynamic
    num_filters_start = 64
    nb_conv = int(np.floor(np.log(output_img_dim[1]) / np.log(2)))
    filters_list = [num_filters_start * min(8, (2 ** i)) for i in range(nb_conv)]

    # CONV 1
    # Do first conv bc it is different from the rest
    # paper skips batch norm for first layer
    disc_out = Convolution2D(nb_filter=64, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name='disc_conv_1')(input_layer)
    disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # CONV 2 - CONV N
    # do the rest of the convs based on the sizes from the filters
    for i, filter_size in enumerate(filters_list[1:]):
        name = 'disc_conv_{}'.format(i+2)

        disc_out = Convolution2D(nb_filter=128, nb_row=4, nb_col=4, border_mode='same', subsample=(stride, stride), name=name)(disc_out)
        disc_out = BatchNormalization(name=name + '_bn', mode=bn_mode, axis=axis)(disc_out)
        disc_out = LeakyReLU(alpha=0.2)(disc_out)

    # ------------------------
    # BUILD PATCH GAN
    # this is where we evaluate the loss over each sublayer of the input
    # ------------------------
    patch_gan_discriminator = generate_patch_gan_loss(last_disc_conv_layer=disc_out,
                                                      patch_dim=patch_dim,
                                                      input_layer=input_layer,
                                                      nb_patches=nb_patches)
    return patch_gan_discriminator


def DCGAN(generator_model, discriminator_model, input_img_dim, patch_dim):
    """
    Here we do the following:
    1. Generate an image with the generator
    2. break up the generated image into patches
    3. feed the patches to a discriminator to get the avg loss across all patches
        (i.e is it fake or not)
    4. the DCGAN outputs the generated image and the loss

    This differs from standard GAN training in that we use patches of the image
    instead of the full image (although a patch size = img_size is basically the whole image)

    :param generator_model:
    :param discriminator_model:
    :param img_dim:
    :param patch_dim:
    :return: DCGAN model
    """

    generator_input = Input(shape=input_img_dim, name="DCGAN_input")

    # generated image model from the generator
    generated_image = generator_model(generator_input)

    h, w = input_img_dim[1:]
    ph, pw = patch_dim

    # chop the generated image into patches
    list_row_idx = [(i * ph, (i + 1) * ph) for i in range(int(h / ph))]
    list_col_idx = [(i * pw, (i + 1) * pw) for i in range(int(w / pw))]

    list_gen_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            x_patch = Lambda(lambda z: z[:, :, row_idx[0]:row_idx[1],
                col_idx[0]:col_idx[1]], output_shape=input_img_dim)(generated_image)
            list_gen_patch.append(x_patch)

    # measure loss from patches of the image (not the actual image)
    dcgan_output = discriminator_model(list_gen_patch)

    # actually turn into keras model
    dc_gan = Model(input=[generator_input], output=[generated_image, dcgan_output], name="DCGAN")
    return dc_gan


def generate_patch_gan_loss(last_disc_conv_layer, patch_dim, input_layer, nb_patches):

    # generate a list of inputs for the different patches to the network
    list_input = [Input(shape=patch_dim, name="patch_gan_input_%s" % i) for i in range(nb_patches)]

    # get an activation
    x_flat = Flatten()(last_disc_conv_layer)
    x = Dense(2, activation='softmax', name="disc_dense")(x_flat)

    patch_gan = Model(input=[input_layer], output=[x, x_flat], name="patch_gan")

    # generate individual losses for each patch
    x = [patch_gan(patch)[0] for patch in list_input]
    x_mbd = [patch_gan(patch)[1] for patch in list_input]

    # merge layers if have multiple patches (aka perceptual loss) 
    if len(x) > 1:
        x = merge(x, mode="concat", name="merged_features")
    else:
        x = x[0]

    # merge mbd if needed
    # mbd = mini batch discrimination
    # https://arxiv.org/pdf/1606.03498.pdf
    if len(x_mbd) > 1:
        x_mbd = merge(x_mbd, mode="concat", name="merged_feature_mbd")
    else:
        x_mbd = x_mbd[0]

    num_kernels = 100
    dim_per_kernel = 5

    M = Dense(num_kernels * dim_per_kernel, bias=False, activation=None)
    MBD = Lambda(minb_disc, output_shape=lambda_output)

    x_mbd = M(x_mbd)
    x_mbd = Reshape((num_kernels, dim_per_kernel))(x_mbd)
    x_mbd = MBD(x_mbd)
    x = merge([x, x_mbd], mode='concat')

    x_out = Dense(2, activation="softmax", name="disc_output")(x)

    discriminator = Model(input=list_input, output=[x_out], name='discriminator_nn')
    return discriminator


def train():
    # ----------------------
    # HYPER PARAMS
    # ----------------------
    # width, height of images to work with. Assumes images are square
    im_width = im_height = 256

    # inpu/oputputt channels in image
    input_channels = 1
    output_channels = 1

    # image dims
    input_img_dim = (input_channels, im_width, im_height)
    output_img_dim = (output_channels, im_width, im_height)

    # We're using PatchGAN setup, so we need the num of non-overlaping patches
    # this is how big we'll make the patches for the discriminator
    # for example. We can break up a 256x256 image in 16 patches of 64x64 each
    sub_patch_dim = (64, 64)
    nb_patch_patches, patch_gan_dim = num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)

    # ----------------------
    # GENERATOR
    # Our generator is an AutoEncoder with U-NET skip connections
    # ----------------------
    generator_nn = UNETGenerator(input_img_dim=input_img_dim, num_output_channels=output_channels)
    generator_nn.summary()

    # ----------------------
    # PATCH GAN DISCRIMINATOR
    # the patch gan averages loss across sub patches of the image
    # it's fancier than the standard gan but produces sharper results
    # ----------------------
    discriminator_nn = PatchGanDiscriminator(output_img_dim=output_img_dim,
                                             patch_dim=patch_gan_dim,
                                             nb_patches=nb_patch_patches)
    discriminator_nn.summary()

    # disable training while we put it through the GAN
    discriminator_nn.trainable = False

    # ------------------------
    # Define Optimizers
    opt_discriminator = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_dcgan = Adam(lr=1E-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # -------------------------
    # compile generator
    generator_nn.compile(loss='mae', optimizer=opt_discriminator)

    # ----------------------
    # MAKE FULL DCGAN
    # ----------------------
    dc_gan_nn = DCGAN(generator_model=generator_nn,
                      discriminator_model=discriminator_nn,
                      input_img_dim=input_img_dim, 
                      patch_dim=sub_patch_dim)

    dc_gan_nn.summary()

    # ---------------------
    # Compile DCGAN
    # we use a combination of mae and bin_crossentropy
    loss = ['mae', 'binary_crossentropy']
    loss_weights = [1E2, 1]
    dc_gan_nn.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    # ---------------------
    # ENABLE DISCRIMINATOR AND COMPILE
    discriminator_nn.trainable = True
    discriminator_nn.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # ------------------------
    # RUN ACTUAL TRAINING
    batch_size = 1
    data_path = WORKING_DIR + '/data/' + DATASET
    nb_epoch = 100
    n_images_per_epoch = 400

    print('Training starting...')
    for epoch in range(0, nb_epoch):

        print('Epoch {}'.format(epoch))
        batch_counter = 1
        start = time.time()
        progbar = keras_generic_utils.Progbar(n_images_per_epoch)

        # init the datasources again for each epoch
        tng_gen = facades_generator(data_dir_name=data_path, data_type='training', im_width=im_width, batch_size=batch_size)
        val_gen = facades_generator(data_dir_name=data_path, data_type='validation', im_width=im_width, batch_size=batch_size)

        # go through 1... n_images_per_epoch (which will go through all buckets as well
        for mini_batch_i in range(0, n_images_per_epoch, batch_size):

            # load a batch of decoded and original images
            # both for training and validation
            X_train_decoded_imgs, X_train_original_imgs = next(tng_gen)
            X_val_decoded_imgs, X_val_original_imgs = next(val_gen)

            # generate a batch of data and feed to the discriminator
            # some images that come out of here are real and some are fake
            # X is image patches for each image in the batch
            # Y is a 1x2 vector for each image. (means fake or not)
            X_discriminator, y_discriminator = get_disc_batch(X_train_original_imgs,
                                                              X_train_decoded_imgs,
                                                              generator_nn,
                                                              batch_counter,
                                                              patch_dim=sub_patch_dim)

            # Update the discriminator
            # print('calculating discriminator loss')
            disc_loss = discriminator_nn.train_on_batch(X_discriminator, y_discriminator)

            # create a batch to feed the generator
            X_gen_target, X_gen = next(gen_batch(X_train_original_imgs, X_train_decoded_imgs, batch_size))
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_nn.trainable = False

            # trainining GAN
            # print('calculating GAN loss...')
            gen_loss = dc_gan_nn.train_on_batch(X_gen, [X_gen_target, y_gen])

            # Unfreeze the discriminator
            discriminator_nn.trainable = True

            # counts batches we've ran through for generating fake vs real images
            batch_counter += 1

            # print losses
            D_log_loss = disc_loss
            gen_total_loss = gen_loss[0].tolist()
            gen_total_loss = min(gen_total_loss, 1000000)
            gen_mae = gen_loss[1].tolist()
            gen_mae = min(gen_mae, 1000000)
            gen_log_loss = gen_loss[2].tolist()
            gen_log_loss = min(gen_log_loss, 1000000)

            progbar.add(batch_size, values=[("Dis logloss", D_log_loss),
                                            ("Gen total", gen_total_loss),
                                            ("Gen L1 (mae)", gen_mae),
                                            ("Gen logloss", gen_log_loss)])

            # ---------------------------
            # Save images for visualization every 2nd batch
            if batch_counter % 2 == 0:

                # print images for training data progress
                plot_generated_batch(X_train_original_imgs, X_train_decoded_imgs, generator_nn, epoch, 'tng', mini_batch_i)

                # print images for validation data
                X_full_val_batch, X_sketch_val_batch = next(gen_batch(X_val_original_imgs, X_val_decoded_imgs, batch_size))
                plot_generated_batch(X_full_val_batch, X_sketch_val_batch, generator_nn, epoch, 'val', mini_batch_i)

        # -----------------------
        # log epoch
        print("")
        print('Epoch %s/%s, Time: %s' % (epoch + 1, nb_epoch, time.time() - start))

        # ------------------------------
        # save weights on every 2nd epoch
        if epoch % 2 == 0:
            gen_weights_path = os.path.join('./pix2pix_out/weights/gen_weights_epoch_%s.h5' % (epoch))
            generator_nn.save_weights(gen_weights_path, overwrite=True)

            disc_weights_path = os.path.join('./pix2pix_out/weights/disc_weights_epoch_%s.h5' % (epoch))
            discriminator_nn.save_weights(disc_weights_path, overwrite=True)

            DCGAN_weights_path = os.path.join('./pix2pix_out/weights/DCGAN_weights_epoch_%s.h5' % (epoch))
            dc_gan_nn.save_weights(DCGAN_weights_path, overwrite=True)


train()
