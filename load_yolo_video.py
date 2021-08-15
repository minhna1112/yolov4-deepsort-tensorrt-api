import tensorrt as trt
import ctypes
import numpy as np
import cv2
import common
import time

from utils import *

import pycuda.driver as cuda


#define some constants
MAX_OUTPUT_BBOX_COUNT = 1000
IOU_THRESH = 0.4
CONF_THRESH = 0.3
INPUT_SHAPE = (3, 416, 416)


print(trt.__version__)

#load yolo layer plugin into TRT Plugin Registry
plugin_path = '../yolov4-tiny-tensorrt_plugin/build/libyolov4plugin.so'
ctypes.CDLL(plugin_path)

#create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class ModelData(object):
    INPUT_NAME = "data"
    INPUT_SHAPE = (3, 416, 416)
    OUTPUT_NAME = "prob"
    OUTPUT_SIZE = 10
    DTYPE = trt.float32

def populate_network(network, weights):


    return None

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
    #im_path = '/home/ivsr/minh/yolov4-tiny-tensorrt/samples/bus.jpg'
    #image = cv2.imread(im_path)
    video_path = "../vid_test.mp4"	
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image_preprocess(frame,  (INPUT_SHAPE[1], INPUT_SHAPE[2]))
        image = image.transpose(2,0, 1)
		# Copy to the pagelocked input buffer
        np.copyto(dst=inputs[0].host, src=image.ravel()) #only 1 input

		#inference
        start = time.time()
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        print(f"Time: {time.time()-start} s")
        start = time.time()
        valid_outputs = nms(output, CONF_THRESH, IOU_THRESH)
        print(f"NMS Time: {time.time()-start} s")
        image = draw_bbox(frame, valid_outputs, network_res = (INPUT_SHAPE[1], INPUT_SHAPE[2]))
        cv2.imshow('out',image)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()
