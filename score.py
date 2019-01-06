import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import onnx
import ngraph as ng
from ngraph_onnx.onnx_importer.importer import import_onnx_model
from benchmark import benchmarking
import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys
import numpy as np
data_dir = os.environ['TESTDATADIR']
assert data_dir is not None, "No data directory"

from models import *

def build_engine(onnx_model='./checkpoint/googLeNet.cinic10.1.onnx', max_batch_size=256):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        # Configure the builder here.
        builder.max_batch_size = max_batch_size
        builder.max_workspace_size = 1 <<  20
        # Parse the model to create a network.
        with open(onnx_model, 'rb') as model:
            parser.parse(model.read())
        # Build and return the engine. Note that the builder, network and parser are destroyed when this function returns.
        return builder.build_cuda_engine(network)

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()
    
    
# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

def do_inference(context, bindings, inputs, outputs, stream, batch_size=1):
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


checkpoint = torch.load('./checkpoint/googLeNet.cinic10.1.pth')
model = GoogLeNet()
model.load_state_dict(checkpoint['net'])

@benchmarking(team=4, task=0, model=model, preprocess_fn=None)
def inference(model, context, inputs, outputs, bindings, stream, testloader,**kwargs):
    total = 0
    correct = 0
    assert kwargs['device'] != None, 'Device error'
    device = kwargs['device']
    
    if device == 'cpu':
        model.to(device)
        model = torch.nn.DataParallel(model)
        model.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets.cuda()).sum().item()
        accuracy = 100.*correct/total
    elif device == 'cuda':
        for batch_idx, (images, targets) in enumerate(testloader):
            images = images.data.numpy()
            targets = targets.data.numpy()
            
            inputs[0].host = images
            pred = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=32)
            pred = np.asarray(pred).reshape(-1,10)
            predicted_label = np.argmax(pred, axis=1)
            predicted_label = predicted_label[:len(targets)]
            correct += np.equal(targets, predicted_label).sum().item()
            total += len(targets)
        accuracy = correct / total * 100.
            
    return accuracy

'''
TESTDATADIR="/tmp/work/data/CINIC-10/test/" python score.py
'''

if __name__=='__main__':
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],  std=[0.24205776, 0.23828046, 0.25874835]),
    ])
    testset = torchvision.datasets.ImageFolder(root=data_dir, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    
    max_batch_size = 128
    engine = build_engine('./checkpoint/googLeNet.cinic10.1.onnx', max_batch_size=max_batch_size)
    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine)
    
    inference(model, context, inputs, outputs, bindings, stream, testloader)