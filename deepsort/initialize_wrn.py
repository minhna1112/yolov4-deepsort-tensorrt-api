from tools import generate_detections as gdet

import numpy as np

frame = np.load('./image.npy')
bboxes = np.load('./bboxes.npy')
#print(frame.shape)
#print(bboxes.shape)
# initialize deep sort
model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)

features = encoder(frame, bboxes)
np.save('features-nano.npy', features)
print(features.shape)