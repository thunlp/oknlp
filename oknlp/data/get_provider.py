def get_provider(device):
    if device == 'cpu':
        providers = [
            'CPUExecutionProvider'
        ] 
    else:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': int(device[-1]),
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }),
            'CPUExecutionProvider',
        ]  
    return providers

    