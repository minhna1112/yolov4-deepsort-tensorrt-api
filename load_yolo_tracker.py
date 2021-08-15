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
plugin_path = './yolov4/yolov4-tiny-tensorrt_plugin/build/libyolov4plugin.so'
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
    #Post process detection result
    out_bboxes, scores, classes, num_objects = get_bboxes_info(valid_outputs)
    #Rescale to x, y, w,h
    out_bboxes = get_resized_bboxes(out_bboxes, num_objects, original_image, network_res=(INPUT_SHAPE[1], INPUT_SHAPE[2]))
    # x,y,w,h to x_min, y_min, w, h
    bboxes = xywh_to_minminwh(out_bboxes)

    # read in all class names from config
    class_names = read_class_names('./coco.names')
        # by default allow all classes in .names file
    allowed_classes = list(class_names.values())
        
        # custom allowed classes (uncomment line below to customize tracker for only people)
    #allowed_classes = ['person']

        # loop through objects and use class index to get class name, allow only classes in allowed_classes list
    names = []
    deleted_indx = []
    for i in range(num_objects):
        class_indx = int(classes[i])
        class_name = class_names[class_indx]
        if class_name not in allowed_classes:
            deleted_indx.append(i)
        else:
            names.append(class_name)
    names = np.array(names)
    count = len(names)

    cv2.putText(original_image, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
    print("Objects being tracked: {}".format(count))
    # delete detections that are not in allowed_classes
    bboxes = np.delete(bboxes, deleted_indx, axis=0)
    scores = np.delete(scores, deleted_indx, axis=0)
    classes = np.delete(classes, deleted_indx, axis=0)
    num_objects = len(bboxes)
    #image = draw_bbox(original_image, valid_outputs, network_res = (INPUT_SHAPE[1], INPUT_SHAPE[2]))
    # store all predictions in one parameter for simplicity when calling functions
    pred_bbox = [bboxes, scores, classes, num_objects]
    image = draw_bbox_tmp(original_image, pred_bbox)
    image = Image.fromarray(image.astype(np.uint8))
    image.show()
    #save_output
    np.save('./image.npy', original_image)
    np.save('./bboxes.npy', bboxes)
    
    
if __name__=="__main__":
    main()
