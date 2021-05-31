from ..config import config
import os
import kara_storage

# download a ud model
def load(task_name, version):
    if len(config.path) == 0:
        raise ValueError("No available download path")
    if 'cuda' in version:
        version = 'gpu'
    task_dir = os.path.join(config.path[0],task_name, version)
    if not os.path.exists(task_dir):
        storage = kara_storage.KaraStorage(config.source)
        storage.load_directory("", task_name, task_dir, version)
    return task_dir
def get_model(m_path):
    return os.path.join(m_path,[x for x in os.listdir(m_path) if '.onnx' in x][0])