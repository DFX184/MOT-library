### some convert functions
import numpy as np
from base_mot import BaseMot
import match.distance as distance
from match.base_assoc import association_with_distance
from tracker.sort_tracker import SortTracker


class SortMot(BaseMot):
    def __init__(self,detecher,min_hit = 3, max_lost  = 5,
                 threshold = 0.3,
                 distance      = "iou"):
        super().__init__(
            detecher,
            max_lost  = max_lost,
            threshold = threshold,
            distance      = distance 
        )
        self.min_hit = min_hit
    
    def add_new_tracker(self,unmatched_detection,boxes):
        for idx in unmatched_detection:
            try:
                self.trackers.append(
                    SortTracker(
                        self.ID,
                        boxes[int(idx)]
                    )
                )
                self.ID += 1
            except:
                print(idx,unmatched_detection)

    
    def get_result(self):
        result = np.array([
            np.append(np.insert(tk.get_state(),0,tk.Id),0) for tk in self.trackers
            if tk.unmatched_time == 0 and tk.time_since_update < 1 and tk.hit_streak >= self.min_hit
        ],dtype = np.float32)        
        if len(result) == 0:
            return np.empty([0,5],dtype = np.float32)
        return result
    
    
    def get_tracker_boxes(self):
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in sorted(to_del,reverse=True):
            self.trackers.pop(t)
            
        return trks