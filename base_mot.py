import numpy as np
import match.distance as distance
from match.base_assoc import association_with_distance
from tracker.base_tracker import BaseTracker


class BaseMot:
    def __init__(self,
                 detector,
                 max_lost  = 5,
                 threshold = 0.3,
                 distance      = "iou"):
        """
        detecher is a function,take a image output boxes and score (eg yolov5,yolovx)
        iou_threshold 
        """
        self.detector         = detector
        self.threshold        = threshold
        self.trackers         = []
        self.ID               = 1
        self.frame            = 0
        self.distance         = distance
        self.max_lost         = max_lost
        
    def add_new_tracker(self,unmatched_detection,boxes):
        for idx in unmatched_detection:
            try:
                self.trackers.append(
                    BaseTracker(
                        self.ID,
                        boxes[idx]
                    )
                )
                self.ID += 1
            except IndexError:
                print(idx)
    
    def assco_func(self,tracker_boxes,boxes):
        return association_with_distance(tracker_boxes,boxes,distance = self.distance,distance_threshold = self.threshold)
    
    def get_result(self):
        result = np.array([
            np.insert(tk.get_state(),0,tk.Id) for tk in self.trackers
            if tk.unmatched_time == 0
        ],dtype = np.float32)
        if len(result) == 0:
            return np.empty([0,5],dtype = np.float32)
        return result
    
    
    def get_tracker_boxes(self):
        return np.array([tk.get_state() for tk in self.trackers if tk.unmatched_time == 0 ],dtype = np.float32)
    
    
    def update(self,img):
        if not(isinstance(img,np.ndarray)):
            raise RuntimeError("image must be numpy array")
            
        self.frame += 1
        boxes      = self.detector(img) # where box is [x1,y1,x2,y2,score]
        boxes      = np.array(boxes,dtype = np.float32)
        
        tracker_boxes = self.get_tracker_boxes()
        
        tracker_boxes = tracker_boxes.reshape(-1,5)
        
        unmatched_detection,unmatched_tracker,matches,to_del = self.assco_func(tracker_boxes,boxes)
                                                                               
        ### association boxes to tracker
        for m in matches:
            self.trackers[m[1]].update(boxes[m[0]])
        ### delete unmatched boxes 
        del_idxs = set()
        
        for idx in unmatched_tracker:
            self.trackers[idx].unmatched_time += 1
            if self.trackers[idx].unmatched_time >= self.max_lost:
                del_idxs.add(
                    idx
                )
        
        del_idxs = list(del_idxs & set(to_del))
        for idx in np.sort(del_idxs)[::-1]:
            try:
                self.trackers.pop(idx)
            except:
                print(idx,len(self.trackers),list(reversed(unmatched_tracker)),unmatched_tracker,len(tracker_boxes),len(boxes))
            
        ### add new boxes
        self.add_new_tracker(unmatched_detection,boxes)
        
        return self.get_result()
