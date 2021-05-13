import os
import torch
import json
import torch.utils.data as Data
from transformers import BertTokenizer
from ..BaseTyping import BaseTyping
from ....nn.models import BertLinearSigmoid
from ....data import load
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from ....utils.torch2engine import get_engine



class Dataset(Data.Dataset):
    def __init__(self, sents, tokenizer, max_length=128):
        self.sents = sents
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sents)

    def __getitem__(self, i):
        [text, span] = self.sents[i]
        if span[0] > self.max_length:
            text = text[span[0]:]
            span = (0, span[1] - span[0])
        text = text[:span[0]] + '<ent>' + text[span[0]: span[1]] + '</ents>' + text[span[1]:]
        pos = [text.index('<ent>')]
        text = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))[:self.max_length]
        text += [0] * (self.max_length - len(text))
        return torch.LongTensor(text), torch.LongTensor(pos)


class BertTyping(BaseTyping):
    """使用Bert模型实现的Typing算法
    """
    def __init__(self, device=None, batch_size=1, max_length=128):
        typ_path = load('typ')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.tokenizer.add_special_tokens({'additional_special_tokens': ["<ent>", "</ent>"]})
        self.types = json.loads(open(os.path.join(typ_path, 'types.json'), 'r').read())
        if self.trt_mode == False:
            self.load_model(typ_path,device)
            self.model.eval()
        else:
            self.load_model(typ_path,device)
            ner_cache = os.path.join(typ_path,'cache')
            if not os.path.exists(ner_cache):
                os.mkdir(ner_cache)
            input_shape = (batch_size, max_length) 
            onnx_file_path = os.path.join(ner_cache,"ner_bert_{}_{}.onnx".format(batch_size, max_length))
            if not os.path.exists(onnx_file_path):
                self.create_onnx(onnx_file_path,input_shape)
            engine_file_path = os.path.join(ner_cache,'ner_bert_{}_{}.trt'.format(batch_size, max_length))
            self.create_context(onnx_file_path,engine_file_path,input_shape)
        self.model.eval()

        super().__init__(device)

    def load_model(self,model_path):
        self.model = BertLinearSigmoid(len(self.tokenizer))
        self.model.load_state_dict(torch.load(os.path.join(model_path, 'typing.pth'), map_location=lambda storage, loc: storage))

        
    def create_onnx(self,onnx_path,input_shape):
        x = torch.ones(*input_shape, dtype = torch.int32)#模型forwad参数shape及类型
        torch.onnx.export(self.model, #pytorch模型
                (x,x),#model调用输入样例
                onnx_path,#输出onnx位置
                opset_version=10#dynimic输入opset版本
                )
    def create_context(self,onnx_path,trt_path,shape):
        self.engine = get_engine(max_batch_size=shape[0],shape=shape,onnx_file_path=onnx_path, engine_file_path =  trt_path)
        self.context = self.engine.create_execution_context() 
        self.d_inputs = [cuda.mem_alloc(self.binding_nbytes(binding)) for binding in self.engine if self.engine.binding_is_input(binding)]
        self.h_output = cuda.pagelocked_empty(tuple(self.engine.get_binding_shape(2)), dtype=np.int32)
        self.d_outputs = cuda.mem_alloc(self.h_output.nbytes)
        self.stream = cuda.Stream() 

    def do_inference(self,context, h_output, inputs, outputs, stream,features):
        cuda.memcpy_htod_async(inputs[0], features[0], stream)
        cuda.memcpy_htod_async(inputs[1], features[1], stream)
        context.execute_async_v2(bindings=[int(d_inp) for d_inp in inputs] + [int(outputs)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output,outputs, stream)
        #[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return h_output

    def binding_nbytes(self,binding):
        return trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize
    
    def to(self, device):
        if self.trt_mode == False:
            self.model = self.model.to(device)
        return super().to(device)

    def __call__(self, sents):
        results = []
        dataset = Dataset(sents, self.tokenizer)
        dataloader = Data.DataLoader(dataset, batch_size=8, num_workers=0)
        for text, pos in dataloader:
            with torch.no_grad():
                if self.trt_mode == True:
                    features = (text.to('cpu').numpy().astype(np.int32),pos.to('cpu').numpy().astype(np.int32))
                    p = self.do_inference(self.context, h_output=self.h_output, inputs=self.d_inputs, outputs=self.d_outputs, stream=self.stream,features=features) # numpy data  
                    p = torch.from_numpy(p).to(self.device)
                    outs = p.long()
                else:
                    outs = self.model(text.to(self.device), pos)
                    outs = outs.cpu().tolist()
            for out in outs:
                result = []
                for i, score in enumerate(out):
                    if score > 0.1:
                        result.append((self.types[i], score))
                results.append(result)
        return results