import os
import numpy as np
import h5py

def normalize(X):
    return X / 255.0

def facades_generator(data_dir_name, data_type, im_width, batch_size=10):
    """
    Generates facades and target images
    X = decoded images
    Y = original images

    :param data_dir_name: Absolute path location of the data folder
    :param data_type: Can be 'training', 'testing', 'validation'
    :param batch_size: Batch size for training
    :return:
    """
    data_dir = data_dir_name + '/' + data_type

    # get a count of buckets in images dir and remove any files that don't have h5 in the name
    bucket_names_in_dir = os.listdir(data_dir + '/images')
    bucket_names_in_dir = [f for f in bucket_names_in_dir if '.h5' in f]

    # iterate forever bc keras requires this
    while True:

        # go through all the buckets
        for file_name in bucket_names_in_dir:
            images_path = data_dir + '/images/' + file_name
            facades_path = data_dir + '/facades/' + file_name

            target_images = h5py.File(images_path, 'r')
            facade_images = h5py.File(facades_path, 'r')

            # go through bucket in batch sizes
            num_images = target_images['data'].shape[0]
            width = height = im_width

            for batch_num in range(0, num_images, batch_size):
                i = batch_num
                i_end = i + batch_size

                # slice the specific batch that we want and output it through the generator
                x_batch_facades = np.array(facade_images['data'][i: i_end], dtype=np.float32)
                x_batch_facades = x_batch_facades.reshape((len(x_batch_facades), 1, width, height))
                x_batch_facades = normalize(x_batch_facades)

                y_batch_images = np.array(target_images['data'][i: i_end], dtype=np.float32)
                y_batch_images = y_batch_images.reshape((len(y_batch_images), 1, width, height))
                y_batch_images = normalize(y_batch_images)

                yield x_batch_facades, y_batch_images