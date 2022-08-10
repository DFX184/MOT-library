from . import distance
from operator import gt,lt
import numpy as np
from scipy.optimize import linear_sum_assignment


_distance_map = {
    "iou" : distance.iou_batch,
    "centroid" : distance.centroid_distance
}

_compare_map = {
    "iou" : (gt,lt),
    "centroid" : (lt,gt),
    "default"  : (lt,gt)
}


def association_with_distance(tracker_boxes,detection_boxes,distance = "iou",distance_threshold = 0.3):
    if len(tracker_boxes) == 0:
        return np.arange(len(detection_boxes)),np.empty((0,5),dtype = np.int_),np.empty((0,2),dtype = np.int_)
    if len(detection_boxes) == 0:
        return np.empty((0,5),dtype = np.int_), np.arange(len(tracker_boxes)),np.empty((0,2),dtype = np.int_)
    
    distance = distance.lower()
    if distance not in _distance_map and str(type(distance)) != "function":
        raise RuntimeError(
            r"Only support `iou`,`centroid`"
        )
    if not(0 < distance_threshold <= 1):
        raise RuntimeError(
            r"distance threshold must be (0,1]"
        )
    if str(type(distance)) == "function":
        matrix = distance(detection_boxes,
                          tracker_boxes)
    else:
        matrix = _distance_map[distance](detection_boxes,tracker_boxes)
    unmatched_tracker = []
    unmatched_detecher= []
    matches = []
    
    comp1,comp2       = _compare_map[distance] if not str(type(distance)) == "function" else _compare_map['default']
    
    
    if min(matrix.shape) > 0:
        a = comp1(matrix,distance_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            
            if distance.lower() == "iou":
                transfunc = lambda x : -1 * x.copy()
            elif distance.lower() == "centroid":
                transfunc = lambda x : x
            x, y = linear_sum_assignment(transfunc(matrix))
            matched_indices = np.array(list(zip(x,y)),dtype = np.int_)
    else:
        matched_indices = np.empty(shape=(0,2))
        
    for idx in range(len(detection_boxes)):
        if idx not in matched_indices[:,0]:
            unmatched_detecher.append(idx)
            
    for idx in range(len(tracker_boxes)):
        if idx not in matched_indices[:,1]:
            unmatched_tracker.append(idx)
    
    if distance != "iou":
        p = int(distance_threshold * 100)
        distance_threshold = np.percentile(matrix,p)
    
    for m in matched_indices:
        if comp2(matrix[m[0],m[1]],distance_threshold):
            unmatched_detecher.append(m[0])
            unmatched_tracker.append(m[1])
        else:
            matches.append(m.reshape(1,2))
            
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0).astype("int")
    
    return (
        unmatched_detecher,
        unmatched_tracker,
        matches
    )
    