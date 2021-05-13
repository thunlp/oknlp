import torch
from ..BaseNER import BaseNER
from ....utils.dataset import Dataset
from ....nn.models import BertSeq as Model
import torch.utils.data as Data
from functools import reduce
from ....utils.format_output import format_output
from ....data import load
from ....config import config
import os
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np
from ....utils.torch2engine import get_engine
from transformers import BertTokenizer
from ....utils.process_io import merge_result, split_text_list

labels = ['O'] + reduce(lambda x,y:x+y, [[f"{kd}-{l}" for kd in ('B','I', 'O')] for l in ('PER','LOC','ORG')])

class BertNER(BaseNER):
    def __init__(self, device=None,batch_size=1,max_length=128):
        ner_path = load('ner')
        self.trt_mode = config.enable_tensorrt
        if device == None:
            device = config.default_device
        if self.trt_mode == False:
            self.load_model(ner_path,device)
            self.model.eval()
        else:
            self.load_model(ner_path,device)
            ner_cache = os.path.join(ner_path,'cache')
            if not os.path.exists(ner_cache):
                os.mkdir(ner_cache)
            input_shape = (batch_size, max_length) 
            onnx_file_path = os.path.join(ner_cache,"ner_bert_{}_{}.onnx".format(batch_size, max_length))
            if not os.path.exists(onnx_file_path):
                self.create_onnx(onnx_file_path,input_shape)
            engine_file_path = os.path.join(ner_cache,'ner_bert_{}_{}.trt'.format(batch_size, max_length))
            self.create_context(onnx_file_path,engine_file_path,input_shape)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        super().__init__(device) 

    def load_model(self,model_path,device):
        self.model = Model()
        self.model.expand_to(len(labels),device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_path,"ner_bert.ckpt"),map_location=torch.device(device)))

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

    def __call__(self,sents):
        self.sents, is_end_list = split_text_list(sents, 126)
        self.test_dataset = Dataset(self.sents, self.tokenizer)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=8, num_workers=0)
        split_ans_list = self.infer_epoch(self.test_loader)
        count = 0
        for i, sent in enumerate(self.sents):
            split_ans = split_ans_list[i]
            for d in split_ans:
                d['begin'] += count
                d['end'] += count
            count += len(sent)
            if is_end_list[i]:
                count = 0
        return merge_result(split_ans_list, is_end_list)

    def infer_step(self, batch):
        x, y, at = batch
        y = y.to(self.device)
        at = at.to(self.device)
        with torch.no_grad():
            if self.trt_mode == True:
                features = (x.to('cpu').numpy().astype(np.int32),at.to('cpu').numpy().astype(np.int32))
                p = self.do_inference(self.context, h_output=self.h_output, inputs=self.d_inputs, outputs=self.d_outputs, stream=self.stream,features=features) # numpy data  
                p = torch.from_numpy(p).to(self.device)
                p = p.long()
            else:
                x = x.to(self.device)
                p = self.model(x, at)
                p = p.to(self.device)
            mask = y != -1
            batch_size = mask.size(0)
            p = p[:batch_size]
        return torch.where(mask, p, 0).cpu().tolist(), mask.cpu().tolist()
    
    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        results =[]
        for i in format_output(self.sents, pred,labels):
            results.append([{'type': j[0], 'begin': j[1], 'end': j[2]} for j in i])
        return results

    
