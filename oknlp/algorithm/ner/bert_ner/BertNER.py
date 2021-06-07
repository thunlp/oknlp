import os
from functools import reduce
from ....utils.format_output import format_output
import kara_storage
import numpy as np
from transformers import BertTokenizer
import onnxruntime as rt
from ..BaseNER import BaseNER
from ....auto_config import get_provider
from ....data import load,get_model
from typing import List, Dict, Union

labels = ['O'] + reduce(lambda x, y: x + y, [[f"{kd}-{l}" for kd in ('B', 'I', 'O')] for l in ('PER', 'LOC', 'ORG')])

class BertNER(BaseNER):
    def __init__(self, device=None, *args, **kwargs):
        self.inited = False
        providers, fp16_mode = get_provider(device)
        if not fp16_mode:
            ner_path = load('ner.bert','cpu')
        else:
            ner_path = load('ner.bert','gpu')
        self.config = {
            "inited": False,
            "ner_path": ner_path,
            "providers": providers
        }
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
            self.sess = rt.InferenceSession(get_model(self.config['ner_path']),sess_options, providers=self.config['providers'])
            self.input_name = self.sess.get_inputs()[0].name
            self.att_name = self.sess.get_inputs()[1].name 
            self.label_name = self.sess.get_outputs()[0].name
            self.config['inited'] = True

        input_feed = {self.input_name: [np.array(i[1]).astype(np.int32) for i in batch], 
            self.att_name: [np.array(i[3]).astype(np.int32) for i in batch]}
        pred_onx = self.sess.run([self.label_name],input_feed)[0]
        mask = np.array([i[2] for i in batch]) != -1
        pred_onx = np.where(mask, pred_onx, -1).tolist() 
        return list(zip([i[0] for i in batch],pred_onx))#合并句子和结果