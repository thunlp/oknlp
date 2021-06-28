import py3nvml

def adaptive_batch_size(gpuid, fp16_mode):
    py3nvml.py3nvml.nvmlInit()
    h = py3nvml.py3nvml.nvmlDeviceGetHandleByIndex(gpuid)
    info = py3nvml.utils.try_get_info(py3nvml.py3nvml.nvmlDeviceGetMemoryInfo, h,
                             ['something'])
    mem_avl = info.free>>30
    if fp16_mode == True:
        batch_size = mem_avl * 16
    else:
        batch_size = mem_avl * 4
    py3nvml.py3nvml.nvmlShutdown()
    batch_size = max(1,batch_size)
    batch_size = min(1024,batch_size)
    return batch_size, mem_avl