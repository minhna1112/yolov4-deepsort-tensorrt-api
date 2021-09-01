import tensorrt as trt
import generate_detections as gdet
import common
import numpy as np

TRT_LOGGER = trt.Logger()
MAX_BATCH_SIZE = 4
frame = np.load('./image.npy')
bboxes = np.load('./bboxes.npy')
engine_path = 'wrn-fc.engine'

encoder = gdet.create_box_encoder(engine_path, TRT_LOGGER, batch_size=1)

features = encoder(frame, bboxes)
#np.save('features-TRT.npy', features)
#print(features.shape)