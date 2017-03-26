import matplotlib.pyplot as plt
import numpy as np

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
