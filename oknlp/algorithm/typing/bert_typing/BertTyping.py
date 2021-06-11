import os
import json
from transformers import BertTokenizer
import numpy as np
import onnxruntime as rt
from ..BaseTyping import BaseTyping
from ....auto_config import get_provider
from ....data import load

class BertTyping(BaseTyping):
    """使用Bert模型实现的Typing算法
    """
    def __init__(self, device=None, *args, **kwargs):
        providers, fp16_mode, batch_size = get_provider(device)
        if not fp16_mode:
            model_path = load('typing.bert','fp32')
        else:
            model_path = load('typing.bert','fp16')
        types = json.loads(open(os.path.join(model_path, 'types.json'), 'r').read())
        self.config = {
            "inited": False,
            "model_path": model_path,
            "providers": providers,
            'types': types
        }
        self.batch_size = batch_size
        super().__init__(*args,**kwargs)
       

    def preprocess(self, x, *args, **kwargs):
        if not self.config['inited']:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.tokenizer.add_special_tokens({'additional_special_tokens': ["<ent>", "</ent>"]})
            self.config['inited'] = True
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
                result.append((self.config['types'][i], float(score)))
        return result

    def inference(self, batch):
        if not self.config['inited']:
            sess_options = rt.SessionOptions()
            sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.sess = rt.InferenceSession(os.path.join(self.config['model_path'],'model.onnx'),sess_options,providers=self.config['providers'])
            self.input_name = self.sess.get_inputs()[0].name
            self.pos = self.sess.get_inputs()[1].name 
            self.att_name = self.sess.get_inputs()[2].name
            self.label_name = self.sess.get_outputs()[0].name
            self.config['inited'] = True
        x, y, at = np.stack(tuple(batch),axis=1)
        max_len = max([len(i) for i in x])
        input_feed = {self.input_name: [np.array(i + [0] * (max_len - len(i))).astype(np.int32) for i in x], 
            self.pos: [np.array(i).astype(np.int64) for i in y], 
            self.att_name: [np.array(i + [0] * (max_len - len(i))).astype(np.int32) for i in at]}
        pred_onx = self.sess.run([self.label_name], input_feed)[0]
        return pred_onx
    