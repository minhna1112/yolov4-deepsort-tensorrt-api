from tensorflow.core.framework.tensor_shape_pb2 import TensorShapeProto
import tensorflow as tf
import graphsurgeon as gs
import numpy as np

HEIGHT =128
WIDTH = 64

graph = gs.DynamicGraph('./deepsort/mars-small128.pb')
image_tensor = graph.find_nodes_by_name('images')

print('Found Input: ', image_tensor)

cast_node = graph.find_nodes_by_name('Cast')[0] #Replace Cast with ToFloat if using tensorflow <1.15
print("Input Field", cast_node.attr['SrcT'])

cast_node.attr['SrcT'].type=1 #Changing Expected type to float
print("Input Field", cast_node.attr['SrcT'])


input_node = gs.create_plugin_node(name='InputNode', op='Placeholder', shape=(-1,HEIGHT,WIDTH,3), dtype=tf.float32)

namespace_plugin_map = {

    'images': input_node
}


graph.collapse_namespaces(namespace_plugin_map)

graph.write('GRAPH_NO_UINT8.pb')
# graph.write_tensorboard('tensorboard_log_modified_graph')