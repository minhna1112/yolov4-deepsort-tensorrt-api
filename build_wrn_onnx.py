import tensorflow.compat.v1 as tf
import numpy as np
import tensorrt as trt
import common
import cv2
#create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
print(trt.__version__)

MAX_BATCH_SIZE = 4
BN_EPSILON = 0.0010000000474974513
BALL_CONSTANT = 9.99999993922529e-09
engine_path = 'wrn-4.engine'

class ModelData(object):
    INPUT_NAME = 'images'
    OUTPUT_NAME = 'features'
    INPUT_SHAPE = (3, 128, 64)
    OUTPUT_SIZE = 128
    DTYPE = trt.float32


def build_engine_from_pb(weights, max_batch_size):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config: 
        builder.max_batch_size = max_batch_size
        network = builder.create_network() 
        config.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_engine(network, config)

def build_engine_from_onnx(onnx_path: str, max_batch_size: int):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config: 
        builder.max_batch_size = max_batch_size
        #network = builder.create_network(1 <<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(1 <<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        config.max_workspace_size = common.GiB(1)
        # Populate the network using ONNX parser
        parser = trt.OnnxParser(network, TRT_LOGGER)
        is_parsed = parser.parse_from_file(onnx_path)
        assert is_parsed
        # Build and return an engine.
        return builder.build_engine(network, config)

engine = build_engine_from_onnx('./wrn_no_unit8-l.onnx', MAX_BATCH_SIZE)

common.serialize_engine_to_file(engine, engine_path)

# # Deserialize saved engine from file
# engine_2 = common.deserialize_engine_from_file(engine_path=engine_path, logger=TRT_LOGGER)
# # Allocate input, output on host memory and GPU device memory
# inputs_2, outputs_2, bindings_2, stream_2 = common.allocate_buffers(engine_2) #Return list of inputs/outputs host_device_memory(buffer) devicebindingsbuffers, cuda stream
# # Create execution context from engine
# context_2 = engine_2.create_execution_context()
# # # Load input array to host memory
# bboxes  = np.random.rand(4, 3, 128, 64).ravel()
# np.copyto(inputs_2[0].host, bboxes)
# # # Do inferences
# [out]= common.do_inference(context_2, bindings=bindings_2, inputs=inputs_2, outputs=outputs_2, stream=stream_2, batch_size=4)

# print(out)