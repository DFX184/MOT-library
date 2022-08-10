import random
class IDColorPool:
    def __init__(self):
        self.color_map = {}
        
    def query(self,Ids):
        return [self.color_map[idx] for idx in Ids]
    
    def random_color(self):
        R = random.randint(0,255)
        G = random.randint(0,255)
        B = random.randint(0,255)
        return (R,G,B)
    def update(self,Ids):
        for idx in Ids:
            if idx not in self.color_map:
                self.color_map[idx] = self.random_color()