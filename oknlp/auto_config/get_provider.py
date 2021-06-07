from .gpu_scheduler import get_gpu_info, get_gpu_utilization, get_gpumem_utilization
import py3nvml

def get_provider(device=None):
    d_cuda = ('CUDAExecutionProvider',{
              'device_id': 0,
              'arena_extend_strategy': 'kNextPowerOfTwo',
              'cudnn_conv_algo_search': 'EXHAUSTIVE'
            })
    d_cpu = 'CPUExecutionProvider'
    fp16_mode = False
    if device==None:
        if get_gpu_info() == []:
            device = 'cpu'
        else:
            gpu_rates = [i['gpu_id']for i in get_gpu_utilization() if i['gpu_rate'] <0.2]
            mem_rates = [i['gpu_id'] for i in get_gpumem_utilization() if i['mem_rate'] >0.7]
            gpu_usable = list(set(gpu_rates).intersection(set(mem_rates)))
            gpu_info = get_gpu_info(gpu_usable)
            fp16_device = [i['gpu_id'] for i in gpu_info if i['gpu_compt']>=7.0]
            #device = 'cuda: {}'.format(fp16_device[0])
    if device == 'cpu':
        providers = [d_cpu]
    elif device != None:
        d_cuda[1]['device_id'] = device[-1]
        if int(device[-1]) in fp16_device:
            fp16_mode = True
        providers = [d_cuda, d_cpu]
    else:
        if fp16_device != []:
            d_cuda[1]['device_id'] = fp16_device[0]
            fp16_mode = True
        else:
            d_cuda[1]['device_id'] = gpu_usable[0]
        providers = [d_cuda, d_cpu]
    return providers, fp16_mode 
