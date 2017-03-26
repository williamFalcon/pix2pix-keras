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
from networks.generator import UNETGenerator
from networks.discriminator import PatchGanDiscriminator
from networks.DCGAN import DCGAN
from utils import patch_utils
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
        y_disc = np.zeros((len(X_disc), 2), dtype=np.uint8)

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
    X_disc = patch_utils.extract_patches(images=X_disc, sub_patch_dim=patch_dim)

    return X_disc, y_disc


def gen_batch(X1, X2, batch_size):

    while True:
        idx = np.random.choice(X1.shape[0], X1.shape[0], replace=False)
        x1 = X1[idx]
        x2 = X2[idx]
        yield x1, x2


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
    nb_patch_patches, patch_gan_dim = patch_utils.num_patches(output_img_dim=output_img_dim, sub_patch_dim=sub_patch_dim)

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
