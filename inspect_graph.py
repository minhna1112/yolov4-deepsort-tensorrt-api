import tensorflow.compat.v1 as tf
import numpy as np

try:
    from tensorflow.contrib.tensorrt.ops.gen_trt_engine_op import *
except ImportError:
    pass

session = tf.Session()
input_name = 'InputNode'
output_name = 'features'

checkpoint_filename = 'GRAPH_NO_UINT8.pb'

output_node_txt = './nodes.txt'
graph_def_txt = './new_graph.txt'

with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(file_handle.read())
    tf.import_graph_def(graph_def, name="net")
    input_var = tf.get_default_graph().get_tensor_by_name("%s:0" % input_name)
    output_var = tf.get_default_graph().get_tensor_by_name("%s:0" % output_name)

    #Print all nodes in graph_def
    #nodes = [n.name + ' => ' + n.op for n in graph_def.node]
    #with open(output_node_txt, 'w') as f:
    #        for node in nodes:
    #                f.write(node+ '\n')

    with open(graph_def_txt, 'w') as f:
            f.write(str(graph_def))
            
     #import to tensorboard
    # from tensorflow.python.summary import summary        
    # pb_visual_writer = summary.FileWriter(log_dir)
    # pb_visual_writer.add_graph(sess.graph)
    # print("Model Imported. Visualize by running: "
    #           "tensorboard --logdir={}".format(log_dir))

    

def get_conv_weights(graph_def, node_name):
    #Get node from node name
    conv_node = [node for node in graph_def.node if node_name == node.name]
    #reight weight content from node name (byte array)
    byte_str = conv_node[0].attr.get('value').tensor.tensor_content
    #weight shape
    weight_shape = conv_node[0].attr.get('value').tensor.tensor_shape
    #REad in to numpy contagious array
    conv_w = np.frombuffer(byte_str, dtype=np.float32)
    #Reshape to correct TRT weight format (#num_fil x C x H x W)
    tf_shape = [weight_shape.dim[i].size for i in range(len(weight_shape.dim))] #C x H x W x #num_fil
    conv_w = np.reshape(conv_w, tf_shape)
    #conv_w = np.transpose(conv_w, (3,0,1,2)) #K, H, W, C
    return conv_w

