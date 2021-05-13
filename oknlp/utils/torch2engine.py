import os
import sys
sys.path.append(os.path.dirname('/root/ink/'))
from ..nn.models import BertSeq as Model
import torch
import onnx
from ..data import load
import tensorrt as trt
from functools import reduce
import logging

TRT_LOGGER = trt.Logger()

logger = logging.Logger(__name__)
def get_engine(max_batch_size=1,shape=(1,128), onnx_file_path='', engine_file_path='',fp16_mode=True, save_engine=True):
    
    """
    params max_batch_size:      预先指定大小好分配显存
    params onnx_file_path:      onnx文件路径
    params engine_file_path:    待保存的序列化的引擎文件路径
    params fp16_mode:           是否采用FP16
    params save_engine:         是否保存引擎
    returns:                    ICudaEngine
    """

    if os.path.exists(engine_file_path):
        logger.info("Reading engine from file: %s" % engine_file_path)
        with open(engine_file_path, 'rb') as f, \
            trt.Runtime(TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())  # 反序列化
    else:  # 由onnx创建cudaEngine
        explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        # In TensorRT 7.0, the ONNX parser only supports full-dimensions mode, meaning that your network definition must be created with the explicitBatch flag set. For more information, see Working With Dynamic Shapes.
    with trt.Builder(TRT_LOGGER) as builder, \
        builder.create_builder_config() as config, \
        builder.create_network(explicit_batch) as network,  \
        trt.OnnxParser(network, TRT_LOGGER) as parser: # 使用onnx的解析器绑定计算图，后续将通过解析填充计算图
        builder.max_batch_size = max_batch_size # 执行时最大可以使用的batchsize
        builder.fp16_mode = fp16_mode
        #dynamic shape设置profile
        profile = builder.create_optimization_profile()
        profile.set_shape("input", shape, shape, shape) 
        profile.set_shape("attention_mask", shape, shape, shape) 
        config.add_optimization_profile(profile)
        config.set_flag(trt.BuilderFlag.FP16)
        config.max_workspace_size = 1<<30#分配workspace
        if not os.path.exists(onnx_file_path):
            quit("ONNX file %s not found!"%onnx_file_path)
        logger.info('loading onnx file from path %s ...'%onnx_file_path)
        with open(onnx_file_path, 'rb') as model: # 二值化的网络结果和参数
            logger.info("Begining onnx file parsing")
            parser.parse(model.read())  # 解析onnx文件
        logger.info("Completed parsing of onnx file")
        logger.info("Building an engine from file %s this may take a while..."%onnx_file_path)
        engine=builder.build_engine(network, config) 
        logger.info("Completed creating Engine")
        if save_engine: 
            with open(engine_file_path, 'wb') as f:
                f.write(engine.serialize())  #序列化写入
        return engine

if __name__=='__main__':
    labels = reduce(lambda x,y:x+y, [[f"{kd}-{l}" for kd in ('B','I','O')] for l in ('SEG',)])
    cws_path = load('cws')
    model = Model()
    batch_size = 1 											
    model.expand_to(len(labels),'cuda:0')
    model.load_state_dict(
        torch.load(os.path.join(cws_path,"cws_bert.ckpt"),map_location=torch.device('cuda:0')))
    input_shape = (128, 128) 				
    x = torch.ones(*input_shape, dtype = torch.int32)#模型forwad参数shape及类型
    export_onnx_file = "cws.bert.onnx" 	
    torch.onnx.export(model, #pytorch模型
                        (x,x),#model调用输入样例
                        export_onnx_file,#输出onnx位置
                        opset_version=10#dynimic输入opset版本
                        )
    get_engine(onnx_file_path='cws.bert.onnx',engine_file_path='cws.bert.engine')