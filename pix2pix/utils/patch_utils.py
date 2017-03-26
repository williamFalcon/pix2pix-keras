import numpy as np


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