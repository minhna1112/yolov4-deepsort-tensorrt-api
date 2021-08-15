import tensorrt as trt
import ctypes
import numpy as np
import cv2
import time
from PIL import Image

import common
from utils import *

import pycuda.driver as cuda

print(trt.__version__)

#load yolo layer plugin into TRT Plugin Registry
plugin_path = '../yolov4-tiny-tensorrt_plugin/build/libyolov4plugin.so'
ctypes.CDLL(plugin_path)

#create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

#define some constants
MAX_OUTPUT_BBOX_COUNT = 1000
IOU_THRESH = 0.4
CONF_THRESH = 0.3
INPUT_SHAPE = (3, 416, 416)

class ModelData(object):
    INPUT_NAME = "data"
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

def add_batchnorm_2D(network: trt.INetworkDefinition, weights: dict, input_tensor: trt.ITensor, layer_name: str, eps: float)->trt.IScaleLayer:
    gamma = weights[f'{layer_name}.weight'].numpy()
    beta = weights[f'{layer_name}.bias'].numpy()
    mean = weights[f'{layer_name}.running_mean'].numpy()
    var = weights[f'{layer_name}.running_var'].numpy()

    scale = trt.Weights(gamma / np.sqrt(var+eps))
    shift = trt.Weights(beta - mean * gamma / np.sqrt(var+eps))
    power = trt.Weights(np.array(np.ones((scale.size,), dtype=np.float32)))

    scale_1 = network.add_scale(input=input_tensor, scale=scale, shift=shift, power=power, mode=trt.ScaleMode.CHANNEL)
    return  scale_1

def conv_bn_leaky(network: trt.INetworkDefinition, weights: dict, input_tensor: trt.ITensor, out_channel: int, ksize: int, s: int, p: int, layer_idx: int):
    #Convolutional layer
    conv1_w = weights[f'module_list.{layer_idx}.Conv2d.weight'].numpy()
    #conv1_b = weights[f'module_list.{layer_idx}.Conv2d.bias'].numpy()
    #conv1 = network.add_convolution_nd(input=input_tensor, kernel=conv1_w, bias=conv1_b,
    #                        num_output_maps=out_channel, kernel_shape=(ksize, ksize))
    conv1 = network.add_convolution_nd(input=input_tensor, kernel=conv1_w, num_output_maps=out_channel, kernel_shape=(ksize, ksize))
    conv1.stride  = (s, s)
    conv1.padding = (p, p)
    #BatchNorm2D
    bn1 = add_batchnorm_2D(network, weights, conv1.get_output(0), f'module_list.{layer_idx}.BatchNorm2d', 1e-4)
    #Activation
    lr = network.add_activation(bn1.get_output(0), trt.ActivationType.LEAKY_RELU)
    return lr

def populate_network(network: trt.INetworkDefinition, weights: dict):
    INPUT_W = ModelData.INPUT_SHAPE[1]
    INPUT_H = ModelData.INPUT_SHAPE[2]

    # Input layer -> (3, In, in)
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)

    #First downsample -> (32, in/2, in/2)
    l0 = conv_bn_leaky(network, weights, input_tensor, 32, 3, 2, 1, 0)

    # Second downsample -> (64, in/4, in/4)
    l1 = conv_bn_leaky(network, weights, l0.get_output(0), 64, 3, 2, 1, 1)


    #BACKBONE
    #First-stage -> (128, in/8, in/8)
    #initial conv
    l2 = conv_bn_leaky(network, weights, l1.get_output(0), 64, 3, 1, 1, 2)
    #split (route_lhalf)
    l3 = network.add_slice(l2.get_output(0), trt.Dims([0,0,0]), trt.Dims([32, int(INPUT_W/4), int(INPUT_W/4)]), trt.Dims([1,1,1]))
    l4 = conv_bn_leaky(network, weights, l3.get_output(0), 32, 3, 1, 1, 4)
    l5 = conv_bn_leaky(network, weights, l4.get_output(0), 32, 3, 1, 1, 5)
    #merge half
    input_tensor_6 = [l5.get_output(0), l4.get_output(0)]
    cat6 = network.add_concatenation(input_tensor_6)
    l7 = conv_bn_leaky(network, weights, cat6.get_output(0), 64, 1, 1, 0, 7)
    #merge all (route all)
    input_tensor_8 = [l2.get_output(0), l7.get_output(0)]
    cat8 = network.add_concatenation(input_tensor_8)
    #transitional pooling
    pool9 = network.add_pooling(cat8.get_output(0), trt.PoolingType.MAX, window_size=(2,2))

    # Second-stage -> (256, in/16, in/16)
    # initial conv
    l10 = conv_bn_leaky(network, weights, pool9.get_output(0), 128, 3, 1, 1, 10)
    # split (route_lhalf)
    l11 = network.add_slice(l10.get_output(0), trt.Dims([0, 0, 0]), trt.Dims([64, int(INPUT_W / 8), int(INPUT_W / 8)]), trt.Dims([1, 1, 1]))
    l12 = conv_bn_leaky(network, weights, l11.get_output(0), 64, 3, 1, 1, 12)
    l13 = conv_bn_leaky(network, weights, l12.get_output(0), 64, 3, 1, 1, 13)
    # merge half
    input_tensor_14 = [l13.get_output(0), l12.get_output(0)]
    cat14 = network.add_concatenation(input_tensor_14)
    l15 = conv_bn_leaky(network, weights, cat14.get_output(0), 128, 1, 1, 0, 15)
    # merge all (route all)
    input_tensor_16 = [l10.get_output(0), l15.get_output(0)]
    cat16 = network.add_concatenation(input_tensor_16)
    # transitional pooling
    pool17 = network.add_pooling(cat16.get_output(0), trt.PoolingType.MAX, window_size=(2, 2))

    # Third-stage -> (512, in/32, in/32)
    # initial conv
    l18 = conv_bn_leaky(network, weights, pool17.get_output(0), 256, 3, 1, 1, 18)
    # split (route_lhalf)
    l19 = network.add_slice(l18.get_output(0), trt.Dims([0,0,0]), trt.Dims([128, int(INPUT_W / 16), int(INPUT_W / 16)]), trt.Dims([1, 1, 1]))
    l20 = conv_bn_leaky(network, weights, l19.get_output(0), 128, 3, 1, 1, 20)
    l21 = conv_bn_leaky(network, weights, l20.get_output(0), 128, 3, 1, 1, 21)
    # merge half
    input_tensor_22 = [l21.get_output(0), l20.get_output(0)]
    cat22 = network.add_concatenation(input_tensor_22)
    l23 = conv_bn_leaky(network, weights, cat22.get_output(0), 256, 1, 1, 0, 23) #For neck 2
    # merge all (route all)
    input_tensor_24 = [l18.get_output(0), l23.get_output(0)]
    cat24 = network.add_concatenation(input_tensor_24)
    # transitional pooling
    pool25 = network.add_pooling(cat24.get_output(0), trt.PoolingType.MAX, window_size=(2, 2))
    #For neck 1
    l26 = conv_bn_leaky(network, weights, pool25.get_output(0), 512, 3, 1, 1, 26)


    #NECK
    #First branch -> (512, in/32, in/32)
    l27 = conv_bn_leaky(network, weights, l26.get_output(0), 256, 1, 1, 1, 27) #For bridge
    l28 = conv_bn_leaky(network, weights, l27.get_output(0), 512, 3, 1, 1, 28)
    #Bridge -> (384, in/16, in/16)
    l31 = l27
    l32 = conv_bn_leaky(network, weights, l31.get_output(0), 128, 1, 1, 1, 32)


    #HEAD
    #H
    l23.get_output(0).name = 'branch_2'
    l28.get_output(0).name = 'branch_1'
    network.mark_output(tensor=l28.get_output(0))
    network.mark_output(tensor=l23.get_output(0))


def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config: 
        network = builder.create_network() 
        config.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_engine(network, config)


def main():
    #pretrained_path = './yolov4-tiny.pt'
    #state_dict = torch.load(pretrained_path)['model']
    #engine = build_engine(state_dict)
    
    engine_path = '../yolov4-tiny-tensorrt/build/yolov4-tiny.engine'
    #common.serialize_engine_to_file(engine, engine_path)
    runtime = trt.Runtime(TRT_LOGGER)
    engine = common.deserialize_engine_from_file(engine_path, runtime)

    #allocate buffers and create a stream.
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    context = engine.create_execution_context()

    #input
    im_path = '/home/ivsr/minh/yolov4-tiny-tensorrt/samples/bus.jpg'
    original_image = cv2.imread(im_path) # Return height * width *3
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = image_preprocess(original_image, (INPUT_SHAPE[1], INPUT_SHAPE[2]))
    image = image.transpose(2,0, 1)
    # Copy to the pagelocked input buffer
    np.copyto(dst=inputs[0].host, src=image.ravel()) #only 1 input

    #inference
    start = time.time()
    [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print(f"Time: {time.time()-start} s")
    start = time.time()
    valid_outputs = nms(output, CONF_THRESH, IOU_THRESH)
    #print(valids_outputs.shape)
    print(f"NMS Time: {time.time()-start} s")
    #visualize
    image = draw_bbox(original_image, valid_outputs, network_res = (INPUT_SHAPE[1], INPUT_SHAPE[2]))
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    #save_output
    
    
if __name__=="__main__":
    main()
