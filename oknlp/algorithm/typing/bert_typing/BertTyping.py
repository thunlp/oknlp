import os
import json
from transformers import BertTokenizerFast
import numpy as np
import onnxruntime as rt
from ..BaseTyping import BaseTyping
from ....auto_config import get_provider
from ....data import load

class BertTyping(BaseTyping):
    """基于BERT的细粒度实体分类算法

    Args:
        device (str): 运行模型设备的名称，例如："cuda:1"，"cpu"。
        batch_size (int): 模型单次推理最大的batch size，默认会根据硬件资源自动设置。
        num_preprocess (int): 预处理函数进程数，默认为一个自动设置的不超过4的值。
        num_postprocess (int): 后处理函数进程数，默认为一个自动设置的不超过4的值。
        max_queue_size (int): 最大调用队列长度，默认为1024.
        multiprocessing_context: 多进程上下文，默认优先使用"fork"方式。
    
    :Name: bert

    **示例**

    .. code-block:: python

        oknlp.typing.get_by_name("bert")
    """

    def __init__(self, device=None, *args, **kwargs):
        provider, provider_op, fp16_mode, batch_size = get_provider(device)
        if not fp16_mode:
            model_path = load('typing.bert','fp32')
        else:
            model_path = load('typing.bert','fp16')
        with open(os.path.join(model_path, 'types.json'), "r", encoding="utf-8") as f_label:
            types = json.loads(f_label.read())
        self.config = {
            "inited": False,
            "model_path": model_path,
            "provider": provider,
            "provider_option": provider_op,
            'types': types,
        }
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)
       
    def init_preprocess(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
        self.tokenizer.add_special_tokens({'additional_special_tokens':["<ent>","</ent>"]})

    def preprocess(self, x, *args, **kwargs):
        text, span = x
        text = text[:span[0]] + '<ent>' + text[span[0]: span[1]] + '</ent>' + text[span[1]:]
        sx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        sy = text.index('<ent>')
        return sx, sy

    def init_postprocess(self):
        self.types = self.config["types"]

    def postprocess(self, x, *args, **kwargs):
        result = [
            (self.types[i], float(x[i])) for i in np.where(x > 0.1)[0]
        ]
        return result

    def init_inference(self):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < os.cpu_count():
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        self.sess = rt.InferenceSession(os.path.join(self.config['model_path'],'model.onnx'),sess_options,
            providers=self.config['provider'], 
            provider_options=self.config["provider_option"])
        self.input_name = self.sess.get_inputs()[0].name
        self.pos_name = self.sess.get_inputs()[1].name 
        self.att_name = self.sess.get_inputs()[2].name
        self.label_name = self.sess.get_outputs()[0].name

    def pack_batch(self, batch):
        max_len = max([len(tokens) for tokens, _ in batch])
        input_array = np.zeros((len(batch), max_len), dtype=np.int32)
        att_array = np.zeros((len(batch), max_len), dtype=np.int32)
        pos_array = np.zeros((len(batch), 1), dtype=np.int64)
        for i, (tokens, position) in enumerate(batch):
            input_array[i, :len(tokens)] = tokens
            pos_array[i, 0] = position
            att_array[i, :len(tokens)] = 1
        return {
            self.input_name: input_array,
            self.pos_name: pos_array,
            self.att_name: att_array,
        }

    def inference(self, input_feed):
        pred_onx = self.sess.run([self.label_name], input_feed)[0]
        return pred_onx
    