import tensorflow.compat.v1 as tf
import numpy as np
import tensorrt as trt
#create logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

session = tf.Session()
input_name = 'images'
output_name = 'features'

checkpoint_filename = './deep_sort/mars-small128.pb'

output_node_txt = './nodes.txt'
graph_def_txt = './graph.txt'

#Read GraphDef from pre-trained protobuf .pb file
with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_handle.read())
    tf.import_graph_def(graph_def, name="net")
    input_var = tf.get_default_graph().get_tensor_by_name("%s:0" % input_name)
    output_var = tf.get_default_graph().get_tensor_by_name("%s:0" % output_name)

def get_weights_fromn_node(graph_def, node_name):
    #Get node from node name
    conv_node = [node for node in graph_def.node if node_name == node.name]
    #reight weight content from node name (byte array)
    byte_str = conv_node[0].attr.get('value').tensor.tensor_content
    #weight shape
    weight_shape = conv_node[0].attr.get('value').tensor.tensor_shape
    #REad in to numpy contagious array
    conv_w = np.frombuffer(byte_str, dtype=np.float32)
    #Reshape to ndarray shape
    tf_shape = [weight_shape.dim[i].size for i in range(len(weight_shape.dim))] #C x H x W x #num_fil
    conv_w = np.reshape(conv_w, tf_shape)
    return conv_w

def get_conv_weights(graph_def, layer_name):
    node_name = f'{layer_name}/weights'
    conv_w = get_weights_fromn_node(graph_def, node_name)
    #Re-order axis: (C, H, W, #filters) -> (#filters, C, H, W)
    conv_w = np.transpose(conv_w, (3,0,1,2))
    return conv_w

def get_bn_weights(graph_def, layer_name):
    scope_name = f'{layer_name}/{layer_name}/bn'
    gamma = None
    beta = None
    mean = None
    var = None
    return None

conv1_w = get_conv_weights(graph_def, "conv1_1")

print(conv1_w.shape)