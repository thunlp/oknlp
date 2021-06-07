import logging
import numpy as np
import os
import py3nvml
import warnings

def get_gpu_info(allocated_gpus=[]):

    # Try connect with NVIDIA drivers
    logger = logging.getLogger(__name__)
    try:
        py3nvml.py3nvml.nvmlInit()
    except:
        str_ = """Couldn't connect to nvml drivers. Check they are installed correctly."""
        warnings.warn(str_, RuntimeWarning)
        logger.warn(str_)
        return []

    num_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()
    if len(allocated_gpus) == 0:
        allocated_gpus = list(range(num_gpus))
    else:
        assert num_gpus > max(allocated_gpus)
    gpu_info = []
    for gpuid in allocated_gpus:
        tmp = {}
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        tmp['gpu_id'] = gpuid
        tmp['gpu_info'] = py3nvml.py3nvml.nvmlDeviceGetName(h)
        tmp['gpu_compt'] = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetCudaComputeCapability, h,
                             ['something'])
        gpu_info.append(tmp)
    py3nvml.py3nvml.nvmlShutdown()
    return gpu_info

def get_gpu_utilization(allocated_gpus=[]):
    '''
    Gets the utilization rates of each gpu
    '''
    # Try connect with NVIDIA drivers
    logger = logging.getLogger(__name__)
    try:
        py3nvml.py3nvml.nvmlInit()
    except:
        str_ = """Couldn't connect to nvml drivers. Check they are installed correctly."""
        warnings.warn(str_, RuntimeWarning)
        logger.warn(str_)
        return []

    num_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()
    if len(allocated_gpus) == 0:
        allocated_gpus = list(range(num_gpus))
    else:
        assert num_gpus > max(allocated_gpus)
    gpu_rates = []
    for gpuid in allocated_gpus:
        tmp = {}
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        rate = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetUtilizationRates, h,
                             ['something'])
        tmp['gpu_id'] = gpuid
        tmp['gpu_rate'] = rate.gpu
        gpu_rates.append(tmp)
    py3nvml.py3nvml.nvmlShutdown()
    return gpu_rates

def get_gpumem_utilization(allocated_gpus=[]):
    '''
    Gets the memory usage of each gpu
    '''
    # Try connect with NVIDIA drivers
    logger = logging.getLogger(__name__)
    try:
        py3nvml.py3nvml.nvmlInit()
    except:
        str_ = """Couldn't connect to nvml drivers. Check they are installed correctly."""
        warnings.warn(str_, RuntimeWarning)
        logger.warn(str_)
        return []
    num_gpus = py3nvml.py3nvml.nvmlDeviceGetCount()
    if len(allocated_gpus) == 0:
        allocated_gpus = list(range(num_gpus))
    else:
        assert num_gpus > max(allocated_gpus)
    mem_rates = []
    for i, gpuid in enumerate(allocated_gpus):
        tmp = {}
        try:
            h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
        except:
            continue
        info = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetMemoryInfo, h,
                             ['something'])
        tmp['gpu_id'] = gpuid
        tmp['mem_rate'] = info.free/info.total
        mem_rates.append(tmp)
    py3nvml.py3nvml.nvmlShutdown()
    return mem_rates

if __name__ == "__main__":
    print(get_gpu_info())
    print(get_gpu_utilization())
    print(get_gpumem_utilization())