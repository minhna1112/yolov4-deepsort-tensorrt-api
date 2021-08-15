import numpy as np
import cv2
import colorsys
import random

def image_preprocess(image, target_size):

    ih, iw    = target_size
    h,  w, _  = image.shape
    #Smart resize
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh), cv2.INTER_CUBIC)
    #Padding
    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    #Normalize
    image_paded = image_paded / 255.
    
    return image_paded

def get_resized_box(bbox, image, network_resolution):
    ih, iw    = network_resolution
    h,  w, _  = image.shape
    
    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)        

    bx, by, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]

    bx = bx - (iw -nw)//2
    by = by - (ih -nh)//2
    
    bx = bx / scale
    by = by / scale
    bw = bw / scale
    bh = bh / scale

    return np.array([bx, by, bw, bh]) 
     

def iou(lbox, rbox):
    inter_box = np.array([max(lbox[0]-lbox[2]/2., rbox[0]-rbox[2]/2.), min(lbox[0]+lbox[2]/2., rbox[0]+rbox[2]/2.), max(lbox[1]-lbox[3]/2., rbox[1]-rbox[3]/2.), min(lbox[1]+lbox[3]/2., rbox[1]+rbox[3]/2.)])
    
    if (inter_box[2] > inter_box[3]) or (inter_box[0] > inter_box[1]):
        return 0.
    interBoxS =(inter_box[1]-inter_box[0])*(inter_box[3]-inter_box[2])
    
    return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS)

def nms(predictions: np.ndarray, conf_threshold: float, iou_threshold: float):
    #Reshape the valid results
    nums_valid = int(predictions[0])
    predictions = predictions[1:(nums_valid*7)+1].reshape(nums_valid, 7)    
    #Filter out low-confidence box
    mask = predictions[:, 4]>conf_threshold
    predictions = predictions[mask]
    
    valids = []
    for class_id in np.unique(predictions[:, 5]):
        #dtypes = [('x', 'float'), ()]
        #Group boxes with the same class into the same group for compare IOU
        valid = predictions[predictions[:, 5]==class_id]
        #Sort boxes in each group w.r.t confidence point
        valid = valid[np.flip(np.argsort(valid[:, 4]))]
        #print(valid)
        accept_idx = []
        delete_idx = []
        for i, v in enumerate(valid):
           if i not in delete_idx: accept_idx.append(i) 
           for j, w in enumerate(valid[i+1:]):
               #IOU = iou(v[:4], w[:4])
               #print(IOU)  
               if iou(v[:4], w[:4]) > iou_threshold: delete_idx.append(i+1+j)
                
        #print(accept_idx)
        #print(delete_idx)
        for i in accept_idx:
            valids.append(valid[i])
     
    return np.vstack(valids)
    #print(len(valids))

def read_class_names(class_file_name):
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def get_bboxes_info(bboxes):
    return bboxes[:, :4], bboxes[:, 4], bboxes[:, 5], len(bboxes)

def get_resized_bboxes(out_boxes, num_boxes, image, network_res):
    for i in range(num_boxes):
        out_boxes[i] = get_resized_box(out_boxes[i], image, network_res)
        #print(out_boxes[i])
    return out_boxes

def xywh_to_minminwh(out_boxes):
    bboxes = np.copy(out_boxes)
    bboxes[:, 0] = out_boxes[:, 0]-out_boxes[:, 2]/2
    bboxes[:, 1] = out_boxes[:, 1]-out_boxes[:, 3]/2
    
    return bboxes

def draw_bbox(image, bboxes, network_res, classes=read_class_names('./coco.names'), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes[:, :4], bboxes[:, 4], bboxes[:, 5], len(bboxes)
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = get_resized_box(out_boxes[i], image, network_res)
        #print(coor)
        x_min = int(coor[0]-coor[2]/2) 
        x_max = int(coor[0]+coor[2]/2)
        y_min = int(coor[1]-coor[3]/2)
        y_max = int(coor[1]+coor[3]/2)
        #print([x_min, x_max, y_min, y_max])
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (x_min, y_min), (x_max, y_max)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image

def draw_bbox_tmp(image, bboxes, classes=read_class_names('./coco.names'), show_label=True):
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    out_boxes, out_scores, out_classes, num_boxes = bboxes[0], bboxes[1], bboxes[2], bboxes[3]
    for i in range(num_boxes):
        if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
        coor = [out_boxes[i, 0],out_boxes[i, 1],out_boxes[i, 2],out_boxes[i, 3]]
        x_min = int(coor[0]) 
        x_max = int(coor[0]+coor[2])
        y_min = int(coor[1])
        y_max = int(coor[1]+coor[3])
        #print([x_min, x_max, y_min, y_max])
        fontScale = 0.5
        score = out_scores[i]
        class_ind = int(out_classes[i])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (x_min, y_min), (x_max, y_max)
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
            c3 = (c1[0] + t_size[0], c1[1] - t_size[1] - 3)
            cv2.rectangle(image, c1, (np.float32(c3[0]), np.float32(c3[1])), bbox_color, -1) #filled

            cv2.putText(image, bbox_mess, (c1[0], np.float32(c1[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
    return image


