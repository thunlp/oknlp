import os
from transformers import BertTokenizer
from .class_list import classlist
from .apply_text_norm import process_sent
from .evaluate_funcs import format_output
import kara_storage
import numpy as np
import onnxruntime as rt

class onnxBertPosTagging:
    """使用Bert模型实现的PosTagging算法
    """

    def __init__(self, *args, **kwargs):
        if not os.path.exists("postagging/postagging.bert.opt.onnx"):
            storage = kara_storage.KaraStorage("https://data.thunlp.org/ink")
            storage.load_directory("", "postagging", "postagging", "gpu")

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'cudnn_conv_algo_search': 'EXHAUSTIVE',
                }),
                'CPUExecutionProvider',
            ]
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession("postagging/postagging.bert.opt.onnx",sess_options)
        self.input_name = self.sess.get_inputs()[0].name
        self.att_name =self.sess.get_inputs()[1].name 
        self.label_name = self.sess.get_outputs()[0].name


    def preprocess(self, x, *args, **kwargs):
        tokens=self.tokenizer.tokenize(x)
        sx = self.tokenizer.convert_tokens_to_ids(['[CLS]'] + tokens + ['[SEP]']) 
        sy = [-1] + [0] * len(tokens) +[-1]
        sat = [1] * (len(tokens) + 2) 
        return np.array(sx).astype(np.int32), np.array(sy).astype(np.int32), np.array(sat).astype(np.int32)

    def postprocess(self, sent, x, *args, **kwargs):
        result = []
        pred, mask = x
        for ((begin, end), tag) in format_output(pred, mask, classlist, dims=1)[1]:
            result.append((sent[begin:end], tag))
        return result

    def inference(self, batch):
        x, y, at = np.stack(tuple(batch),axis=1)
        mask = y != -1
        input_feed = {self.input_name: x, self.att_name: at}
        pred_onx = self.sess.run([self.label_name], input_feed)[0]
        result = []
        for i in range(len(mask)):
            tmp = (np.where(mask, pred_onx, -1)[i],mask[i])
            result.append(tmp)
        return result

    