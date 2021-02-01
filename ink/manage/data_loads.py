"""
utilities for getting resources
"""
from ..config import config
import os
import requests
import zipfile

from tqdm import tqdm

# set home dir for default


DEFAULT_DOWNLOAD_VERSION = 'latest'

basic_data_needs = ['sgns300','rev_vocab','vocab']
derfault_nlp_missions =['seq','ner','cws','typ']
def get_source_dir():
    return os.path.join(config.path[0], 'sources')
def get_model_url():
    return config.source

# download a ud models zip file
def download_ud_model(task_name, resource_dir=None, should_unzip=True, confirm_if_exists=False, force=False,
                      version=DEFAULT_DOWNLOAD_VERSION):
    if resource_dir is not None and os.path.exists(os.path.join(resource_dir, "{}_models".format(task_name))):
        if confirm_if_exists:
            print("")
            print("The model directory already exists at \"{}/{}_models\". Do you want to download the models again? [y/N]".format(resource_dir,task_name))
            should_download = 'y' if force else input()
            should_download = should_download.strip().lower() in ['yes', 'y']
        else:
            should_download = False
    else:
        print('Would you like to download the models for: '+task_name+' now? (Y/n)')
        should_download = 'y' if force else input()
        should_download = should_download.strip().lower() in ['yes', 'y', '']
    if should_download:
        # set up data directory
        if resource_dir is None:
            print('')
            print('Default download directory: ' + get_source_dir())
            print('Hit enter to continue or type an alternate directory.')
            where_to_download = '' if force else input()
            if where_to_download != '':
                download_dir = where_to_download
            else:
                download_dir = get_source_dir()
        else:
            download_dir = resource_dir
        if not os.path.exists(download_dir):
            os.makedirs(download_dir)
        print('')
        print('Downloading models for: '+task_name)
        model_zip_file_name = task_name+'.zip'
        download_url = get_model_url()+model_zip_file_name
        print(download_url)
        download_file_path = os.path.join(download_dir, model_zip_file_name)
        print('Download location: '+download_file_path)

        # initiate download
        r = requests.get(download_url, stream=True)
        with open(download_file_path, 'wb') as f:
            file_size = int(r.headers.get('content-length'))
            default_chunk_size = 67108864
            with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=default_chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
        # unzip models file
        print('')
        print('Download complete.  Models saved to: '+download_file_path)
        if should_unzip:
            unzip_ud_model(task_name, download_file_path, download_dir)
        # remove the zipe file
        print("Cleaning up...", end="")
        os.remove(download_file_path)
        print('Done.')


# unzip a ud models zip file
def unzip_ud_model(lang_name, zip_file_src, zip_file_target):
    print('Extracting models file for: '+lang_name)
    with zipfile.ZipFile(zip_file_src, "r") as zip_ref:
        zip_ref.extractall(zip_file_target)


# main download function
def loads(download_label, resource_dir=get_source_dir(), confirm_if_exists=False, force=False, version=DEFAULT_DOWNLOAD_VERSION):
    if download_label in basic_data_needs:
        print('downloading basic data needs')
        download_ud_model(download_label, resource_dir=resource_dir, confirm_if_exists=confirm_if_exists, force=force,
                          version=version)

    elif download_label in derfault_nlp_missions:
        print('downloading {} data'.format(download_label))
        download_ud_model(download_label, resource_dir=resource_dir,
                          confirm_if_exists=confirm_if_exists, force=force, version=version)
    else:
        raise ValueError('The data of '+download_label+'is not currently supported by this function. Please try again with other name.')




