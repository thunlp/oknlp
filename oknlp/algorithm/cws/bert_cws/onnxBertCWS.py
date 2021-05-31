import os
from functools import reduce
from ....utils.format_output import format_output
from transformers import BertTokenizer
import kara_storage
import numpy as np
import onnxruntime as rt
from ....data import get_provider,load,get_model

labels = reduce(lambda x,y:x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])
class onnxBertCWS:
    def __init__(self, device = None, *args, **kwargs):
        if device == None:
            device = 'cpu'
        cws_path = load('cws.bert', device)
        providers = get_provider(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(get_model(cws_path),sess_options, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.att_name =self.sess.get_inputs()[1].name 
        self.label_name = self.sess.get_outputs()[0].name


    def preprocess(self, x, *args, **kwargs):
        tokens=self.tokenizer.tokenize(x)
        sx = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) 
        sy = [-1] + [0] * len(tokens) +[-1]
        sat = [1] * (len(tokens) + 2) 
        return x, sx, sy, sat

    def postprocess(self, x, *args, **kwargs):
        return [x[0][j[1]:j[2] + 1] for j in format_output(x[1], labels + ['O']) if x[0][j[1]:j[2] + 1]]

    def inference(self, batch):
        input_feed = {self.input_name: [np.array(i[1]).astype(np.int32) for i in batch], 
            self.att_name: [np.array(i[3]).astype(np.int32) for i in batch]}
        pred_onx = self.sess.run([self.label_name],input_feed)[0]
        mask = [i[2] for i in batch] != -1
        pred_onx = np.where(mask, pred_onx, -1).tolist() 
        return list(zip([i[0] for i in batch],pred_onx))#合并句子和结果