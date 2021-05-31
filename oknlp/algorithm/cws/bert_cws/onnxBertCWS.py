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
        return np.array(sx).astype(np.int32), np.array(sy).astype(np.int32), np.array(sat).astype(np.int32)

    def postprocess(self, x, pred, *args, **kwargs):
        return [x[j[1]:j[2] + 1] for j in format_output(x, pred, labels + ['O'])]

    def inference(self, batch):
        x, y, at = np.stack(tuple(batch),axis=1)
        mask = y != -1
        input_feed = {self.input_name: x, self.att_name: at}
        pred_onx = self.sess.run(
            [self.label_name],input_feed)[0]
        return np.where(mask, pred_onx, -1).tolist()