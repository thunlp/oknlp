from ..config import config
import os
import kara_storage


def load(task_name, version):
    if len(config.path) == 0:
        raise ValueError("No available download path")
    task_dir = os.path.join(config.path, task_name, version)
    if not os.path.exists(task_dir):
        storage = kara_storage.KaraStorage(config.source)
        storage.load_directory("", task_name, task_dir, version)
    return task_dir