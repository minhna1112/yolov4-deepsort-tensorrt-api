import tensorflow.compat.v1 as tf
import numpy as np
import tensorrt as trt
import common

#create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
print(trt.__version__)

session = tf.Session()
input_name = 'images'
output_name = 'features'

checkpoint_filename = './deepsort/mars-small128.pb'

output_node_txt = './nodes.txt'
graph_def_txt = './graph.txt'

class ModelData(object):
    INPUT_NAME = 'images'
    OUTPUT_NAME = 'features'
    INPUT_SHAPE = (3, 128, 64)
    OUTPUT_SIZE = 128
    DTYPE = trt.float32

#Read GraphDef from pre-trained protobuf .pb file
with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_handle.read())
    tf.import_graph_def(graph_def, name="net")
    input_var = tf.get_default_graph().get_tensor_by_name("%s:0" % input_name)
    output_var = tf.get_default_graph().get_tensor_by_name("%s:0" % output_name)

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
    print(conv_w.shape)
    return conv_w

def get_conv_weights(graph_def, layer_name: str):
    node_name = f'{layer_name}/weights'
    conv_w = get_weights_from_node(graph_def, node_name)
    #Re-order axis: (C, H, W, #filters) -> (#filters, C, H, W)
    conv_w = np.transpose(conv_w, (3,2,0,1))
    return conv_w

def get_bn_weights(graph_def, layer_name: str):
    scope_name = f'{layer_name}/bn'
    beta = get_weights_from_node(graph_def, scope_name+'/beta')
    mean = get_weights_from_node(graph_def, scope_name+'/moving_mean')
    var = get_weights_from_node(graph_def, scope_name+'/moving_variance')
    return beta, mean, var


def add_batchnorm_2D(network: trt.INetworkDefinition, graph_def, input_tensor: trt.ITensor, layer_name: str, eps: float)->trt.IScaleLayer:
    #gamma = weights[f'{layer_name}.weight'].numpy()
    beta, mean, var = get_bn_weights(graph_def, layer_name)
    gamma = 1.0
    scale = trt.Weights(gamma / np.sqrt(var+eps))
    shift = trt.Weights(beta - mean * gamma / np.sqrt(var+eps))
    power = trt.Weights(np.array(np.ones((scale.size,), dtype=np.float32)))

    scale_1 = network.add_scale(input=input_tensor, scale=scale, shift=shift, power=power, mode=trt.ScaleMode.CHANNEL)
    return  scale_1

def conv_bn_elu(network: trt.INetworkDefinition, graph_def, input_tensor: trt.ITensor, out_channel: int, ksize: int, s: int, p: int, layer_name: str):
    #Convolutional layer
    conv1_w = get_conv_weights(graph_def, layer_name)
    conv1 = network.add_convolution_nd(input=input_tensor, kernel=conv1_w, num_output_maps=out_channel, kernel_shape=(ksize, ksize))
    conv1.stride  = (s, s)
    conv1.padding = (p, p)
    #BatchNorm2D
    bn1 = add_batchnorm_2D(network, graph_def, conv1.get_output(0), f'{layer_name}/{layer_name}', 0.0010000000474974513)
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
        bn1 = add_batchnorm_2D(network, graph_def, input_tensor, block_name, 0.0010000000474974513)
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


def populate_network(network: trt.INetworkDefinition, graph_def):
    INPUT_W = ModelData.INPUT_SHAPE[1]
    INPUT_H = ModelData.INPUT_SHAPE[2]
    # Input layer -> (3, 128, 64)
    input_tensor = network.add_input(name=ModelData.INPUT_NAME, dtype=ModelData.DTYPE, shape=ModelData.INPUT_SHAPE)
    # (3, 128, 64) -> (32, 128, 64)
    conv1_1 = conv_bn_elu(network, graph_def, input_tensor, 32, 3, 1, 1, 'conv1_1')
    # (3, 128, 64) -> (32, 128, 64)
    conv1_2 = conv_bn_elu(network, graph_def, conv1_1.get_output(0), 32, 3, 1, 1, 'conv1_2')
    # transitional pooling: # (3, 128, 64) -> (32, 64, 32)
    pool = network.add_pooling(conv1_2.get_output(0), trt.PoolingType.MAX, window_size=(2, 2))
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

    network.mark_output(tensor=res_9.get_output(0))

def build_engine(weights):
    # For more information on TRT basics, refer to the introductory samples.
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config: 
        network = builder.create_network() 
        config.max_workspace_size = common.GiB(1)
        # Populate the network using weights from the PyTorch model.
        populate_network(network, weights)
        # Build and return an engine.
        return builder.build_engine(network, config)

# conv1_w = get_conv_weights(graph_def, "conv1_2")
# print(conv1_w.shape)
# bn1_w = get_bn_weights(graph_def, "conv1_1")

engine = build_engine(graph_def)