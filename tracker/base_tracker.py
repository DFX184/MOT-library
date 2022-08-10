
class BaseTracker(object):
    def __init__(self,Id,box = None):
        self.Id  = Id
        self.box = box
        self.unmatched_time = 0
        
    def get_state(self):
        return self.box
    
    def update(self,box):
        self.box = box
        self.unmatched_time = 0
        