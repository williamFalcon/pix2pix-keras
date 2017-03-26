from keras.utils.data_utils import get_file
import os
import subprocess


# ------------------------
# CONSTANTS
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
print(ROOT_DIR)
WORKING_DIR = os.path.dirname(os.path.realpath(__file__))
AWS_FACADES_PATH = 'https://s3.amazonaws.com/open-source-william-falcon/facades_bw.tar'


# -----------------------
# MAIN FUNCTION
def download():
    ds_name = 'facades_bw'

    # make tmp dir
    tmp_path = './.tmp'
    print('making tmp dir... {}'.format(WORKING_DIR + '/.tmp'))
    mk_tmp_dirs(tmp_path)

    # make data dir
    data_folder_path = './data'
    print('making data dir... {}'.format(WORKING_DIR + '/data'))
    mk_data_dirs(data_folder_path, ds_name)

    download_facades_bw(WORKING_DIR + '/../.tmp', data_folder_path)


def download_facades_bw(tmp_path, data_folder_path):
    # download to .tmp file
    downloaded_path = get_file(tmp_path + '/facades_bw.tar', origin=AWS_FACADES_PATH)

    # un-tar
    untar_file(downloaded_path, data_folder_path + '/facades_bw', remove_tar=False, flags='-xvf')

    # move data file
    subprocess.call(['rm', '-rf', tmp_path])


def mk_tmp_dirs(dir_name):
    """
    Tar files will be downloaded here while the shuffling process happens
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)


def mk_data_dirs(dir_name, dataset_name):
    """
    Final place where data will be stored
    :param dir_name:
    :return:
    """
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)

    if not os.path.isdir(dir_name + '/' + dataset_name):
        os.mkdir('{}/{}'.format(dir_name, dataset_name))


def untar_file(source_path, destination_path, remove_tar=False, flags='-xf'):
    subprocess.call(['tar', flags, source_path, '-C', destination_path])
    if remove_tar:
        subprocess.call(['rm', source_path])


if __name__ == '__main__':
    download()