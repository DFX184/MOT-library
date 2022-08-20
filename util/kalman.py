import numpy as np

class Kalmanfilter(object):
    def __init__(self,A,H,Q,R,P,prior = None,B=None):
        self.A = A
        self.Q = Q
        self.R = R
        self.B = B
        self.P = P
        self.H = H
        self.prior = None if prior is None else np.array(prior)
        
    def predict(self,mu=None):
        if self.prior is None:
            return None
        x_k = self.A @ self.prior
        if not self.B is None and not mu is None:
            x_k += self.B @ mu
        return x_k
    
    def update(self,x):
        if self.prior is None:
            self.prior= x
            return
        x_k    = self.predict()
        P_k    = self.A @ self.P @ self.A.T + self.Q
        ### Correction
        
        Kk     = P_k @ self.H @ np.linalg.inv(self.H @ P_k @ self.H.T + self.R)
        tmp        = Kk @ self.H 
        self.P     = (np.eye(tmp.shape[0]) - tmp) @ P_k
        self.prior = x_k + Kk @ (x - self.H @ x_k)
        
        
if __name__ == "__main__":
    pass
