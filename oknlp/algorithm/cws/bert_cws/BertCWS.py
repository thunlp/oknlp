import os
from functools import reduce
from ....utils.format_output import format_output
from transformers import BertTokenizer
import numpy as np
import onnxruntime as rt
from ..BaseCWS import BaseCWS
from ....auto_config import get_provider
from ....data import load
import time

labels = reduce(lambda x, y: x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])
class BertCWS(BaseCWS):
    def __init__(self, device = None, *args, **kwargs):

        providers, fp16_mode, batch_size = get_provider(device)
        if not fp16_mode:
            model_path = load('cws.bert','fp32')
        else:
            model_path = load('cws.bert','fp16')
        self.config = {
            "inited": False,
            "model_path": model_path,
            "providers": providers
        }
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)

    def preprocess(self, x, *args, **kwargs):
        if not self.config['inited']:
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.config['inited'] = True
        tokens = self.tokenizer.tokenize(x)
        sx = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) 
        sy = [-1] + [0] * len(tokens) +[-1]
        sat = [1] * (len(tokens) + 2) 
        return x, sx, sy, sat

    def postprocess(self, x, *args, **kwargs):
        return [x[0][j[1]:j[2] + 1] for j in format_output(x[1], labels + ['O']) if x[0][j[1]:j[2] + 1]]

    def inference(self, batch):
        if not self.config['inited']:
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < os.cpu_count():
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
            self.sess = rt.InferenceSession(os.path.join(self.config['model_path'],'model.onnx'),sess_options, providers=self.config['providers'])
            self.input_name = self.sess.get_inputs()[0].name
            self.att_name = self.sess.get_inputs()[1].name 
            self.label_name = self.sess.get_outputs()[0].name
            self.config['inited'] = True
        
        max_len = max([len(i[1]) for i in batch])
        input_array = [np.array(i[1] + [0] * (max_len - len(i[1]))).astype(np.int32) for i in batch]
        att_array = [np.array(i[3] + [0] * (max_len - len(i[3]))).astype(np.int32) for i in batch]
        input_feed = {self.input_name: input_array, self.att_name: att_array }

        pred_onx = self.sess.run([self.label_name],input_feed)[0]
        pred_onx = [i[0][:len(i[1])] for i in list(zip(pred_onx, [i[2] for i in batch]))]
        return list(zip([i[0] for i in batch],pred_onx))#合并句子和结果