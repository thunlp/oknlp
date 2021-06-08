import re
from .gpu_scheduler import get_gpu_info, get_gpu_utilization, get_gpumem_utilization

def get_provider(device=None):
    d_cuda = ('CUDAExecutionProvider',{
              'device_id': 0,
              'arena_extend_strategy': 'kNextPowerOfTwo',
              'cudnn_conv_algo_search': 'EXHAUSTIVE'
            })
    d_cpu = 'CPUExecutionProvider'
    fp16_mode = False

    comp_usable = [i['gpu_id'] for i in get_gpu_utilization() if i['gpu_rate'] <0.2]
    mem_usable = [i['gpu_id'] for i in get_gpumem_utilization() if i['mem_rate'] >0.7]
    gpu_usable = list(set(comp_usable).intersection(set(mem_usable)))
    gpu_info = get_gpu_info(gpu_usable)
    fp16_device = [i['gpu_id'] for i in gpu_info if i['gpu_compt']>=6.1]

    if device==None:
        if get_gpu_info() == []:
            device = 'cpu'
        else:
            if len(fp16_device) !=0:
                device = 'cuda: {}'.format(fp16_device[0])
            elif len(gpu_usable) !=0:
                device = 'cuda: {}'.format(gpu_usable[0])
            else:
                device = 'cpu'

    if device == 'cpu':
        providers = [d_cpu]
    elif 'cuda' in device:
        pattern = re.compile(r'\d+')    
        span = re.search(pattern, device).span() 
        d_cuda[1]['device_id'] = int(device[span[0]:span[1]])
        if d_cuda[1]['device_id'] in fp16_device:
            fp16_mode = True
        providers = [d_cuda, d_cpu]
    return providers, fp16_mode 

