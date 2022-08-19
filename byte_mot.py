import numpy as np
from match.base_assoc import match_box,_compare_map
import sort_mot

def association_with_byte(tracker_boxes,detection_boxes,high_score_threshold = 0.6,lower_score_threshold = 0.2,**kargs):
    if "distance" not in kargs:
        distance  = "iou"
    else:
        distance  = kargs["distance"]
    
    if "distance_threshold" not in kargs:
        distance_threshold = kargs['distance_threshold']
    else:
        distance_threshold = 0.3
    
    if len(tracker_boxes) == 0:
        return np.arange(len(detection_boxes)),np.empty((0,5),dtype = np.int_),np.empty((0,2),dtype = np.int_),[]
    if len(detection_boxes) == 0:
        return np.empty((0,5),dtype = np.int_), np.arange(len(tracker_boxes)),np.empty((0,2),dtype = np.int_),[]
    
    high_detecher = []
    high_index    = []
    lower_detecher= []
    lower_index   = []
    to_del        = []
    for idx,box in enumerate(detection_boxes):
        
        if box[-1] > high_score_threshold:
            high_detecher.append(box)
            high_index.append(idx)
        elif lower_score_threshold <= box[-1] <= high_score_threshold :
            lower_detecher.append(box)
            lower_index.append(idx)
            
    matches       = []
    high_detecher = np.array(high_detecher,dtype = np.float32)
    lower_detecher= np.array(lower_detecher,dtype= np.float32)
    lower_index   = np.array(lower_index,dtype = np.int_)
    high_index    = np.array(high_index,dtype  = np.int_)

    ## first matches (high boxes)
    
    matches_idx_high,high_score_matrix = match_box(tracker_boxes,high_detecher,distance,distance_threshold)
    
    unmatched_tracker_index_high       = { idx for idx in range(len(tracker_boxes)) if idx not in matches_idx_high[:,1] }
    unmatched_detection_high           = { high_index[idx] for idx in range(len(high_detecher)) if idx not in matches_idx_high[:,0] }
    
    comp1,comp2       = _compare_map[distance] if not str(type(distance)) == "function" else _compare_map['default']
    
    for m in matches_idx_high:
        if comp2(high_score_matrix[m[0],m[1]],distance_threshold):
            unmatched_detection_high.add(high_index[m[0]])
            unmatched_tracker_index_high.add(m[1])
        else:
            res = np.array(
                [high_index[m[0]],m[1]]
            )
            matches.append(res.reshape(1,2))
            
    ## second matches (lower boxes)
    unmatched_detection_high = list(unmatched_detection_high)
    unmatched_tracker_index_high = list(unmatched_tracker_index_high)
    
    matches_idx_lower,lower_score_matrix = match_box(tracker_boxes[unmatched_tracker_index_high],lower_detecher,distance,distance_threshold)
    lower_tracker_idx                    = unmatched_tracker_index_high
    
    ## unmatched lower
     
    unmatched_tracker_index_lower      = [ lower_tracker_idx[idx]           
                                            for idx in range(len(lower_tracker_idx)) if idx not in matches_idx_lower[:,1] ]
    unmatched_detection_lower          = [ lower_index[idx]                  
                                          for idx in range(len(lower_detecher))    if idx not in matches_idx_lower[:,0] ]
    for m in matches_idx_lower:
        if comp2(lower_score_matrix[m[0],m[1]],distance_threshold):
            unmatched_detection_lower.append(lower_index[m[0]])
            unmatched_tracker_index_lower.append(lower_tracker_idx[m[1]])
        else:
            res = np.array([[lower_index[m[0]],lower_tracker_idx[m[1]]]])
            matches.append(res.reshape(1,2))
    
    unmatched_tracker_index_lower = list(unmatched_tracker_index_lower)
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.array(matches)
        matches = np.concatenate(matches,axis=0).astype("int")
    
    unmatched_detection_high = np.array(unmatched_detection_high,dtype = np.int_)
    unmatched_detection_lower= np.array(unmatched_detection_lower,dtype = np.int_)
    unmatch_detection = np.concatenate([unmatched_detection_high,unmatched_detection_lower],dtype = np.int_)
    to_del            = unmatched_tracker_index_lower
    unmatch_trackers  = list(set(unmatched_tracker_index_high) - set(unmatched_tracker_index_lower))
    unmatch_trackers  = np.array(unmatch_trackers)
    
    if len(unmatch_trackers) == 0:
        unmatch_trackers = np.empty((0,5),dtype = np.int_)
    if len(unmatch_detection) == 0:
        unmatch_detection = np.empty((0,5),dtype = np.int_)
    return (
            unmatch_detection,
            unmatch_trackers,
            matches,
            to_del
    )


class ByteMot(sort_mot.SortMot):
    def __init__(self,detehcer,
                 high_score_threshold = 0.6,
                 lower_score_threshold= 0.2,
                 min_hit = 3, max_lost  = 5,
                 threshold = 0.3,
                 distance      = "iou"):
        super().__init__(
            detecher = detehcer,
            min_hit  = min_hit,
            max_lost = max_lost,
            threshold= threshold,
            distance = "iou"
        )
        self.high_score_threshold = high_score_threshold
        self.lower_score_threshold= lower_score_threshold
        
    def assco_func(self,tracker_boxes,boxes):
        return association_with_byte(tracker_boxes,boxes,
                                     self.high_score_threshold,
                                     self.lower_score_threshold,
                                     distance= self.distance,
                                     distance_threshold = self.threshold)
    