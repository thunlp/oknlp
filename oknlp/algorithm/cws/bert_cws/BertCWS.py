import os
from functools import reduce
from ....utils.format_output import format_output
from transformers import BertTokenizerFast
import numpy as np
import onnxruntime as rt
from ..BaseCWS import BaseCWS
from ....auto_config import get_provider
from ....data import load
import time

labels = reduce(lambda x, y: x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])
class BertCWS(BaseCWS):
    def __init__(self, device = None, *args, **kwargs):

        provider, provider_op, fp16_mode, batch_size = get_provider(device)
        if not fp16_mode:
            model_path = load('cws.bert','fp32')
        else:
            model_path = load('cws.bert','fp16')
        self.config = {
            "model_path": model_path,
            "provider": provider,
            "provider_option": provider_op,
        }
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = batch_size
        super().__init__(*args, **kwargs)

    def init_preprocess(self):
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-chinese")

    def preprocess(self, x, *args, **kwargs):
        tokens = self.tokenizer.tokenize(x)
        sx = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) 
        return x, sx

    def postprocess(self, x, *args, **kwargs):
        return [x[0][j[1]:j[2] + 1] for j in format_output(x[1], labels + ['O']) if x[0][j[1]:j[2] + 1]]

    def init_inference(self):
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        if hasattr(os, "sched_getaffinity") and len(os.sched_getaffinity(0)) < os.cpu_count():
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
        self.sess = rt.InferenceSession(
            os.path.join(self.config['model_path'],'model.onnx'),
            sess_options, 
            providers=self.config['provider'], 
            provider_options=self.config["provider_option"])
        self.input_name = self.sess.get_inputs()[0].name
        self.att_name = self.sess.get_inputs()[1].name 
        self.label_name = self.sess.get_outputs()[0].name

    def pack_batch(self, batch):
        max_len = max([len(i[1]) for i in batch])
        input_array = np.zeros((len(batch), max_len), dtype=np.int32)
        att_array = np.zeros((len(batch), max_len), dtype=np.int32)

        new_batch = []
        for i, (sent, tokens) in enumerate(batch):
            input_array[i, :len(tokens)] = tokens
            att_array[i, :len(tokens)] = 1

            new_batch.append((sent, len(tokens)))
        
        input_feed = {self.input_name: input_array, self.att_name: att_array }
        return new_batch, input_feed

    def inference(self, batch):
        sent_list, input_feed = batch
        pred_onx = self.sess.run([self.label_name],input_feed)[0]

        pred_onx = [
            (sent, pred[:length])
            for pred, (sent, length) in zip(pred_onx, sent_list)
        ]
        return pred_onx