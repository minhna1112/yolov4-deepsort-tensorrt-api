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
engine_path = 'wrn-fc.engine'

class ModelData(object):
    INPUT_NAME = 'images'
    OUTPUT_NAME = 'features'
    INPUT_SHAPE = (3, 128, 64)
    OUTPUT_SIZE = 128
    DTYPE = trt.float32

def get_weights_from_node(graph_def, node_name: str):
    #Get node from node name
    conv_node = [node for node in graph_def.node if node_name == node.name]
    #reight weight content from node name (byte array)
    byte_str = conv_node[0].attr.get('value').tensor.tensor_content
    #weight shape
    weight_shape = conv_node[0].attr.get('value').tensor.tensor_shape
    #REad in to numpy contagious array
    conv_w = np.frombuffer(byte_str, dtype=np.float32)
    #Reshape to ndarray shape
    tf_shape = [weight_shape.dim[i].size for i in range(len(weight_shape.dim))] #H x W x C x #num_fil
    conv_w = np.reshape(conv_w, tf_shape)
    #print(f'{node_name} -> {conv_w.shape}')
    return conv_w

def get_conv_weights(graph_def, layer_name: str):
    node_name = f'{layer_name}/weights'
    conv_w = get_weights_from_node(graph_def, node_name)
    #Re-order axis: (C, H, W, #filters) -> (#filters, C, H, W)
    conv_w = np.transpose(conv_w, (3,2,0,1))
    return trt.Weights(np.ascontiguousarray(conv_w))

def get_bn_weights(graph_def, layer_name: str):
    scope_name = f'{layer_name}/bn'
    beta = get_weights_from_node(graph_def, scope_name+'/beta')
    mean = get_weights_from_node(graph_def, scope_name+'/moving_mean')
    var = get_weights_from_node(graph_def, scope_name+'/moving_variance')
    return beta, mean, var

def add_ball_bn(network: trt.INetworkDefinition, graph_def, input_tensor: trt.ITensor, eps: float, mode=trt.ScaleMode.ELEMENTWISE)->trt.IScaleLayer:
    beta = get_weights_from_node(graph_def, 'ball/beta')
    mean = get_weights_from_node(graph_def, 'ball/moving_mean')
    var = get_weights_from_node(graph_def, 'ball/moving_variance')
    gamma = 1.0
    scale = trt.Weights(gamma / np.sqrt(var+eps))
    shift = trt.Weights(beta - mean * gamma / np.sqrt(var+eps))
    power = trt.Weights(np.array(np.ones((scale.size,), dtype=np.float32)))

    scale_1 = network.add_scale(input=input_tensor, scale=scale, shift=shift, power=power, mode=mode)
    return  scale_1

def add_batchnorm_2D(network: trt.INetworkDefinition, graph_def, input_tensor: trt.ITensor, layer_name: str, eps: float, mode=trt.ScaleMode.CHANNEL)->trt.IScaleLayer:
    #gamma = weights[f'{layer_name}.weight'].numpy()
    beta, mean, var = get_bn_weights(graph_def, layer_name)
    gamma = 1.0
    scale = trt.Weights(gamma / np.sqrt(var+eps))
    shift = trt.Weights(beta - mean * gamma / np.sqrt(var+eps))
    power = trt.Weights(np.array(np.ones((scale.size,), dtype=np.float32)))

    scale_1 = network.add_scale(input=input_tensor, scale=scale, shift=shift, power=power, mode=mode)
    return  scale_1

def conv_bn_elu(network: trt.INetworkDefinition, graph_def, input_tensor: trt.ITensor, out_channel: int, ksize: int, s: int, p: int, layer_name: str):
    #Convolutional layer
    conv1_w = get_conv_weights(graph_def, layer_name)
    conv1 = network.add_convolution_nd(input=input_tensor, kernel=conv1_w, num_output_maps=out_channel, kernel_shape=(ksize, ksize))
    conv1.stride  = (s, s)
    conv1.padding = (p, p)
    #BatchNorm2D
    bn1 = add_batchnorm_2D(network, graph_def, conv1.get_output(0), f'{layer_name}/{layer_name}', BN_EPSILON)
    #Activation
    elu = network.add_activation(bn1.get_output(0), trt.ActivationType.ELU)
    return elu

def residual_block(network: trt.INetworkDefinition, graph_def, input_tensor: trt.ITensor, is_first: bool, down_sample: bool, out_channel: int, block_name: str):
    #Identity route
    if down_sample:
        s = 2
        proj_w = get_conv_weights(graph_def, f'{block_name}/projection')
        proj = network.add_convolution_nd(input=input_tensor, kernel=proj_w, num_output_maps=out_channel, kernel_shape=(1, 1))
        proj.stride  = (s, s)
        proj.padding = (0, 0)
        identity = proj.get_output(0)
    else:
        s=1
        identity = input_tensor

    #conv route
    if is_first:
        incoming = input_tensor
        block_name = 'conv2_1'
    else:
        bn1 = add_batchnorm_2D(network, graph_def, input_tensor, block_name, BN_EPSILON)
        elu_1 = network.add_activation(bn1.get_output(0), trt.ActivationType.ELU)
        incoming = elu_1.get_output(0)
    #1st conv + bn+ activation
    conv_1 = conv_bn_elu(network, graph_def, incoming, out_channel, 3, s, 1, f'{block_name}/1')
    #2nd conv with bias add
    conv_3_w = get_conv_weights(graph_def, f'{block_name}/2')
    conv_3_b = get_weights_from_node(graph_def, f'{block_name}/2/biases')
    conv_3 = network.add_convolution_nd(input=conv_1.get_output(0), kernel=conv_3_w, bias=conv_3_b, num_output_maps=out_channel, kernel_shape=(3, 3))
    conv_3.stride  = (1, 1)
    conv_3.padding = (1, 1)
    
    #Add2 route
    add_1 = network.add_elementwise(identity, conv_3.get_output(0), trt.ElementWiseOperation.SUM)

    return add_1

def l2_normalization_layer(network: trt.INetworkDefinition, input_tensor: trt.ITensor, eps: float):
    # (128,1, 1) -> (128,)
    squeezed = network.add_shuffle(input_tensor)
    squeezed.reshape_dims = trt.Dims([128,])
    # (f0, f1, ... fn) -> (f0^2 + f1^2 + ... fn^2) (scalar)
    dot_prod = network.add_matrix_multiply(input0=squeezed.get_output(0), op0=trt.MatrixOperation.VECTOR, 
                                        input1=squeezed.get_output(0), op1=trt.MatrixOperation.VECTOR)
    #Unsqueeze to tensor rank 3
    unsqueezed = network.add_shuffle(dot_prod.get_output(0))
    unsqueezed.reshape_dims = trt.Dims([1,1,1])
    # Add non-zero epsilon constant
    add_6 = network.add_scale(unsqueezed.get_output(0), 
                        shift=trt.Weights(np.ascontiguousarray(eps, dtype=np.float32)), 
                        mode=trt.ScaleMode.UNIFORM)
    #Calculated norm sqrt[(f0^2 + f1^2 + ... fn^2)]
    norm = network.add_unary(add_6.get_output(0), op=trt.UnaryOperation.SQRT)    
    
    return norm

def populate_network(network: trt.INetworkDefinition, graph_def):
    INPUT_W = ModelData.INPUT_SHAPE[1]
    INPUT_H = ModelData.INPUT_SHAPE[2]
    # Input layer -> (3, 128, 64)
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    # (3, 128, 64) -> (32, 128, 64)
    conv1_1 = conv_bn_elu(network, graph_def, input_tensor, 32, 3, 1, 1, 'conv1_1')
    # (3, 128, 64) -> (32, 128, 64)
    conv1_2 = conv_bn_elu(network, graph_def, conv1_1.get_output(0), 32, 3, 1, 1, 'conv1_2')
    # transitional pooling: # (32, 128, 64) -> (32, 64, 32)
    pool = network.add_pooling(conv1_2.get_output(0), trt.PoolingType.MAX, window_size=(3, 3))
    pool.stride = (2,2)
    pool.padding = (1,1)
    #1st residual layer: (32, 64, 32) -> (32, 64, 32)
    res_4 = residual_block(network, graph_def, pool.get_output(0), is_first=True, down_sample=False, out_channel=32, block_name='conv2_1')
    #2nd residual layer (32, 64, 32) -> (32, 64, 32)
    res_5 = residual_block(network, graph_def, res_4.get_output(0), is_first=False, down_sample=False, out_channel=32, block_name='conv2_3')
    #3rd residual layer (32, 64, 32) -> (64, 32, 16)
    res_6 = residual_block(network, graph_def, res_5.get_output(0), is_first=False, down_sample=True, out_channel=64, block_name='conv3_1')
    #4th residual layer (64, 32, 16) -> (64, 32, 16)
    res_7 = residual_block(network, graph_def, res_6.get_output(0), is_first=False, down_sample=False, out_channel=64, block_name='conv3_3')
    #5th residual layer (64, 32, 16) -> (128, 16, 8)
    res_8 = residual_block(network, graph_def, res_7.get_output(0), is_first=False, down_sample=True, out_channel=128, block_name='conv4_1')
    #6th residual layer (64, 32, 16) -> (128, 16, 8)
    res_9 = residual_block(network, graph_def, res_8.get_output(0), is_first=False, down_sample=False, out_channel=128, block_name='conv4_3')
    #Reshuffle (Re-permute) to H, W, C for tf.MatMul:  (128, 16, 8) ->(16, 8, 128)-> 16384
    shuffle = network.add_shuffle(res_9.get_output(0))
    shuffle.first_transpose = trt.Permutation([1,2,0])
    #Densely connected: . fc = matmul(x, W.t)  -> 128
    fc_w = get_weights_from_node(graph_def, 'fc1/weights').T #Weights shape(16384, 128) -> (128, 16384) (KxX row-major order)
    fc_10 = network.add_fully_connected(input=shuffle.get_output(0), num_outputs=128, kernel=fc_w)
    fc_bn = add_batchnorm_2D(network, graph_def, fc_10.get_output(0), 'fc1/fc1', BN_EPSILON, trt.ScaleMode.CHANNEL)
    fc_elu = network.add_activation(fc_bn.get_output(0), trt.ActivationType.ELU)
    #l2-norm Unit Hypersphere
    ball_bn = add_ball_bn(network, graph_def, fc_elu.get_output(0), BN_EPSILON, trt.ScaleMode.CHANNEL)
    #Calculated l2-norm features output
    denominator = l2_normalization_layer(network, ball_bn.get_output(0), BALL_CONSTANT)
    nominator = ball_bn
    out = network.add_elementwise(nominator.get_output(0), denominator.get_output(0),  trt.ElementWiseOperation.DIV)
    network.mark_output(tensor=out.get_output(0))
    

def build_engine(weights, max_batch_size):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config: 
        builder.max_batch_size = max_batch_size
        network = builder.create_network() 
        config.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_engine(network, config)


# session = tf.Session()
# input_name = 'images'
# output_name = 'features'

# checkpoint_filename = './deepsort/mars-small128.pb'

# output_node_txt = './nodes.txt'
# graph_def_txt = './graph.txt'

# #Read GraphDef from pre-trained protobuf .pb file
# with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
#     graph_def = tf.GraphDef()
#     graph_def.ParseFromString(file_handle.read())
#     tf.import_graph_def(graph_def, name="net")
#     input_var = tf.get_default_graph().get_tensor_by_name("%s:0" % input_name)
#     output_var = tf.get_default_graph().get_tensor_by_name("%s:0" % output_name)

# engine = build_engine(graph_def, MAX_BATCH_SIZE)

# common.serialize_engine_to_file(engine, engine_path)

# Deserialize saved engine from file
engine_2 = common.deserialize_engine_from_file(engine_path=engine_path, logger=TRT_LOGGER)
# Allocate input, output on host memory and GPU device memory
inputs_2, outputs_2, bindings_2, stream_2 = common.allocate_buffers(engine_2) #Return list of inputs/outputs host_device_memory(buffer) devicebindingsbuffers, cuda stream
# Create execution context from engine
context_2 = engine_2.create_execution_context()
# # Load input array to host memory
bboxes  = np.random.rand(4, 3, 128, 64).ravel()
np.copyto(inputs_2[0].host, bboxes)
# # Do inferences
[out]= common.do_inference(context_2, bindings=bindings_2, inputs=inputs_2, outputs=outputs_2, stream=stream_2, batch_size=4)

