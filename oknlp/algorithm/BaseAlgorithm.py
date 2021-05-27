from ..config import config


class BaseAlgorithm:
    
    """算法类的基类，派生类需要实现preprocess,inference,postprocess函数,
    分别执行数据的预处理，推断，生成指定格式的结果
    
    """
    def __init__(self, num_preprocess=None, num_postprocess=None, batch_size=None,
*args, **kwargs):
        pass
    def preprocess(self, x, *args, **kwargs):
        pass
    def postprocess(self, x, *args, **kwargs):
        pass
    def inference(self, batch):
        pass
