import os
from functools import reduce
from ....utils.format_output import format_output
import numpy as np
from transformers import BertTokenizer
import onnxruntime as rt
from ..BaseNER import BaseNER
from ....auto_config import get_provider
from ....data import load

labels = ['O'] + reduce(lambda x, y: x + y, [[f"{kd}-{l}" for kd in ('B', 'I', 'O')] for l in ('PER', 'LOC', 'ORG')])

class BertNER(BaseNER):
    def __init__(self, device=None, *args, **kwargs):
        provider, provider_op, fp16_mode, batch_size = get_provider(device)
        if not fp16_mode:
            model_path = load('ner.bert','fp32')
        else:
            model_path = load('ner.bert','fp16')
        self.config = {
            "inited": False,
            "model_path": model_path,
            "provider": provider,
            "provider_option": provider_op
        }
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = batch_size
        super().__init__(*args,**kwargs)
        

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
        sent, pred = x
        return [{'type': j[0], 'begin': j[1], 'end': j[2]} for j in format_output(pred, labels)]
        
    def inference(self, batch):
        if not self.config['inited']:
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < os.cpu_count():
                sess_options.intra_op_num_threads = 1
                sess_options.inter_op_num_threads = 1
            self.sess = rt.InferenceSession(os.path.join(self.config['model_path'], 'model.onnx'),sess_options, 
                providers=self.config['provider'], 
                provider_options=self.config["provider_option"])
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