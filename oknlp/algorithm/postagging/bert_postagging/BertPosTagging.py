import os
from transformers import BertTokenizer
from .class_list import classlist
from .apply_text_norm import process_sent
from .evaluate_funcs import format_output
import kara_storage
import numpy as np
import onnxruntime as rt
from ..BasePosTagging import BasePosTagging
from ....auto_config import get_provider
from ....data import load,get_model

class BertPosTagging(BasePosTagging):
    """使用Bert模型实现的PosTagging算法
    """

    def __init__(self, device=None, *args, **kwargs):
        self.inited = False
        providers, fp16_mode = get_provider(device)
        if not fp16_mode:
            postagging_path = load('postagging.bert','cpu')
        else:
            postagging_path = load('postagging.bert','gpu')
        self.config = {
            "inited": False,
            "postagging_path": postagging_path,
            "providers": providers
        }
        super().__init__(*args,**kwargs)


    def preprocess(self, x, *args, **kwargs):
        if not self.config['inited']:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.config['inited'] = True
        tokens=self.tokenizer.tokenize(x)
        sx = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) 
        sy = [-1] + [0] * len(tokens) +[-1]
        sat = [1] * (len(tokens) + 2) 
        return x, np.array(sx).astype(np.int32), np.array(sy).astype(np.int32), np.array(sat).astype(np.int32)

    def postprocess(self, x, *args, **kwargs):
        result = []
        sent, (pred, mask) = x
        for ((begin, end), tag) in format_output(pred, mask, classlist, dims=1)[1]:
            result.append((sent[begin:end], tag))
        return result

    def inference(self, batch):
        if not self.config['inited']:
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = rt.InferenceSession(get_model(self.config['postagging_path']),sess_options, providers=self.config['providers'])
            self.input_name = self.sess.get_inputs()[0].name
            self.att_name = self.sess.get_inputs()[1].name 
            self.label_name = self.sess.get_outputs()[0].name
            self.config['inited'] = True
        input_feed = {self.input_name: [np.array(i[1]).astype(np.int32) for i in batch], 
            self.att_name: [np.array(i[3]).astype(np.int32) for i in batch]}
        pred_onx = self.sess.run([self.label_name],input_feed)[0]
        mask = np.array([i[2] for i in batch]) != -1
        return list(zip([i[0] for i in batch],list(zip(np.where(mask, np.where(mask, pred_onx, -1).tolist(), -1),mask))))
