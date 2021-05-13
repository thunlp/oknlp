#--*-- coding:utf-8 --*--
import pycuda.autoinit 
import pycuda.driver as cuda
import tensorrt as trt
import time 
import torch
from functools import reduce
import torch.utils.data as Data
from ink.utils.dataset import Dataset
import numpy as np
from ink.algorithm.cws.BaseCWS import BaseCWS
from ink.algorithm.cws import BertCWS
import os
from ink.utils.format_output import format_output
from torch2engine import get_engine
labels = reduce(lambda x,y:x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])

class trtBertCWS(BaseCWS):
    def __init__(self,trt_mode=False,device=None):
        self.cws_path = load('cws')
        self.trt_mode = trt_mode
        if device == None:
            device = config.default_device
        if trt_mode == False:
            self.model = Model()
            self.model.expand_to(len(labels),device)
            self.model.load_state_dict(
                torch.load(os.path.join(self.cws_path,"cws_bert.ckpt"),map_location=torch.device(device)))
            self.model.eval()
        else:
            trt_engine_path = "cws.bert.engine"
            self.engine = get_engine(max_batch_size, trt_engine_path)
            self.context = self.engine.create_execution_context() 
            #self.input_features = sents_pre_processor(sents)
            self.d_inputs = [cuda.mem_alloc(self.binding_nbytes(binding)) for binding in self.engine if self.engine.binding_is_input(binding)]
            self.h_output = cuda.pagelocked_empty(tuple(self.engine.get_binding_shape(2)), dtype=np.int32)
            self.d_outputs = cuda.mem_alloc(self.h_output.nbytes)
            self.stream = cuda.Stream()  
        ssuper().__init__(device) 
    def do_inference(self,context, h_output, inputs, outputs, stream,features, batch_size=1):
        cuda.memcpy_htod_async(inputs[0], features[0], stream)
        cuda.memcpy_htod_async(inputs[1], features[1], stream)
        context.execute_async_v2(bindings=[int(d_inp) for d_inp in inputs] + [int(outputs)], stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output,outputs, stream)
        #[cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        stream.synchronize()
        return h_output
    def to(self, device):
        if self.trt_mode == False:
            self.model = self.model.to(device)
        return super().to(device)
    def binding_nbytes(self,binding):
        return trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize

    def __call__(self,sents):
        self.sents = sents
        self.test_dataset = Dataset(self.sents)
        self.test_loader = Data.DataLoader(self.test_dataset, batch_size=10, num_workers=0)
        return self.infer_epoch(self.test_loader)    

    def infer_step(self, batch):
        x, y, at = batch
        y = y.to(self.device)
        at = at.to(self.device)
        with torch.no_grad():
            if self.trt_mode == True:
                features = (x[0].to('cpu').numpy().astype(np.int32),at[0].to('cpu').numpy().astype(np.int32))
                p = self.do_inference(self.context, h_output=self.h_output, inputs=self.d_inputs, outputs=self.d_outputs, stream=self.stream,features=features) # numpy data  
                p = torch.from_numpy(p).to(self.device)
                p = p.long()
            else:
                x = x.to(self.device)
                p = self.model(x, at)
                p = p.to(self.device)
            mask = y != -1
        return torch.where(mask, p, -1).cpu().tolist(), mask.cpu().tolist()

    def infer_epoch(self, infer_loader):
        pred, mask = [], []
        for batch in infer_loader:
            p, m = self.infer_step(batch)
            pred += p
            mask += m
        results =[]
        for i in range(len(self.sents)):
            tmp = format_output(self.sents, pred, labels+['O'])[i]
            results.append([self.sents[i][j[1]:j[2]+1] for j in tmp])
        return results

if __name__ =='__main__':
    import pandas as pd
    from tqdm import tqdm
    dt = pd.read_csv('test.txt',nrows=500)
    sents =dt['sentence'].tolist()
    trt_cws = trtBertCWS(trt_mode=True,device='cuda:0')
    #cws = BertCWS('cuda:0')
    t1 = time.time()
    results = trt_cws(sents)
    # for sent in tqdm(sents):
    #     tmp = []
    #     tmp.append(sent)
    #     results = trt_cws(tmp)
    t2 = time.time()
    print(t2-t1)
    t3 = time.time()
    results = cws(sents)    
    t4 = time.time()
    print(t4-t3)
   