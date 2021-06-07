import os
import json
from transformers import BertTokenizer
import kara_storage
import numpy as np
import onnxruntime as rt
from ..BaseTyping import BaseTyping
from ....auto_config import get_provider
from ....data import load,get_model

class BertTyping(BaseTyping):
    """使用Bert模型实现的Typing算法
    """
    def __init__(self, device=None, *args, **kwargs):
        super().__init__(*args,**kwargs)
        providers, fp16_mode = get_provider(device)
        if not fp16_mode:
            typing_path = load('typing.bert','cpu')
        else:
            typing_path = load('typing.bert','gpu')
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<ent>", "</ent>"]})
        self.types = json.loads(open(os.path.join(typing_path, 'types.json'), 'r').read())
        sess_options = rt.SessionOptions()
        sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = rt.InferenceSession(get_model(typing_path),sess_options,providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.pos = self.sess.get_inputs()[1].name 
        self.att_name = self.sess.get_inputs()[2].name
        self.label_name = self.sess.get_outputs()[0].name

    def preprocess(self, x, *args, **kwargs):
        text, span = x
        text = text[:span[0]] + '<ent>' + text[span[0]: span[1]] + '</ent>' + text[span[1]:]
        sx = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        sy = [text.index('<ent>')]
        sat = [1] * (len(sx)) 
        return sx, sy , sat


    def postprocess(self, x, *args, **kwargs):
        result = []
        for i, score in enumerate(x):
            if float(score) > 0.1:
                result.append((self.types[i], score))
        return result

    def inference(self, batch):
        x, y, at = np.stack(tuple(batch),axis=1)
        input_feed = {self.input_name: [np.array(i).astype(np.int32) for i in x], 
            self.pos: [np.array(i).astype(np.int64) for i in y], 
            self.att_name: [np.array(i).astype(np.int32) for i in at]}
        pred_onx = self.sess.run([self.label_name], input_feed)[0]
        return pred_onx
    