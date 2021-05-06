from ..config import config
import os
import requests
import zipfile
import logging
from tqdm import tqdm
from .resources import get_resouce_info

logger = logging.Logger(__name__)

def check_file_comp(files, resource_dir):
    for fl in files:
        if not os.path.exists(os.path.join(resource_dir, fl)):
            return False
    return True

# download a ud models zip file
def download_ud_model(task_name, version=None):
    if len(config.path) == 0:
        raise ValueError("No available download path")
    try:
        info = get_resouce_info(task_name, version=version)
    except ValueError:
        raise ValueError("Resource name `%s` not found" % task_name)


    download_dir = None
    for path in config.path:
        task_path = os.path.join( os.path.abspath(path), "sources", task_name)
        if os.path.exists(task_path) and check_file_comp(info["files"], task_path):
            download_dir = task_path
            break

    if download_dir is None:
        download_dir = os.path.join( os.path.abspath(config.path[0]), "sources", task_name)
        os.makedirs(download_dir, exist_ok=True)

        download_url = config.source + info["path"]
        download_file_path = os.path.join(download_dir, "resource.zip")

        logger.info('Download location: %s' % download_file_path)
        # initiate download
        r = requests.get(download_url, stream=True)
        with open(download_file_path, 'wb') as f:
            file_size = int(r.headers.get('content-length'))
            default_chunk_size = 1024 * 512
            with tqdm(total=file_size, unit='B', desc="Downloading %s" % task_name, unit_scale=True) as pbar:
                for chunk in r.iter_content(chunk_size=default_chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        pbar.update(len(chunk))
        # unzip models file
        logger.info('Download complete.  Models saved to: %s' % download_file_path)
        unzip_ud_model(task_name, download_file_path, download_dir)
        # remove the zipe file
        logger.info("Cleaning up...")
        os.remove(download_file_path)
        if check_file_comp(info["files"], download_dir):
            logger.info('Done.')
        else:
            logger.warning('Incomplete data may cause problems with subsequent model loading.')
    else:
        logger.info('Data already exists.')
    return download_dir

# unzip a ud models zip file
def unzip_ud_model(task_name, zip_file_src, zip_file_target):
    logger.info('Extracting models file for: %s' % task_name)
    with zipfile.ZipFile(zip_file_src, "r") as zip_ref:
        zip_ref.extractall(zip_file_target)


# main download function
def load(download_label):
    df_path =  download_ud_model(download_label)
    return df_path




