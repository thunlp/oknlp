"""
utilities for getting resources
"""
from ..config import config
import os
import requests
import zipfile
import logging
from tqdm import tqdm
from ink.data.resources import get_resouce_info

# set home dir for default
logger = logging.Logger(__name__)


basic_data_needs = ['sgns300','rev_vocab','vocab']
default_nlp_missions =['seq','cws','ner','typ']

def get_data_dir(task_name):
    assert  len(config.path) != 0
    source_dir = os.path.join(config.path[0], 'sources')
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
    if task_name in default_nlp_missions:
        download_dir = os.path.join(source_dir, task_name)
    else:
        download_dir = os.path.join(source_dir, 'basic_data')
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    return  download_dir

def get_model_url():
    return config.source

def check_file_comp(task_name,resource_dir):
    for fl in get_resouce_info(task_name)['files']:
        if not os.path.exists(os.path.join(resource_dir, fl)):
            return False
    return True

# download a ud models zip file
def download_ud_model(task_name):
    download_dir = get_data_dir(task_name)
    if not check_file_comp(task_name,download_dir):
        logger.info('Downloading models for: '+task_name)
        model_zip_file_name = task_name+'.zip'
        download_url = get_model_url()+model_zip_file_name
        logger.info(download_url)
        download_file_path = os.path.join(download_dir, model_zip_file_name)
        logger.info('Download location: '+download_file_path)
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
        logger.info('Download complete.  Models saved to: '+download_file_path)
        unzip_ud_model(task_name, download_file_path, download_dir)
        # remove the zipe file
        logger.info("Cleaning up...")
        os.remove(download_file_path)
        if check_file_comp(task_name,download_dir):
            logger.info('Done.')
        else:
            logger.warning('Incomplete data may cause problems with subsequent model loading.')
    else:
        logger.info('Data already exists.')
    return download_dir

# unzip a ud models zip file
def unzip_ud_model(task_name, zip_file_src, zip_file_target):
    logger.info('Extracting models file for: '+task_name)
    with zipfile.ZipFile(zip_file_src, "r") as zip_ref:
        zip_ref.extractall(zip_file_target)


# main download function
def load(download_label):
    if download_label in basic_data_needs or download_label in default_nlp_missions:
        df_path =  download_ud_model(download_label)
        return df_path
    elif download_label == 'basic':
        for bta in basic_data_needs:
            df_path = download_ud_model(bta)
        return df_path
    else:
        raise ValueError('The data of %s is not currently supported by this function. Please try again with other name.'%download_label)




