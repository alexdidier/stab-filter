'''
Copyright (c) 2023, ETH Zurich, Alexandre Didier, Andrea Zanelli, Kim P. Wabersich, Prof. Dr. Melanie N. Zeilinger, 
Institute for Dynamic Systems and Control, D-MAVT
'''

import numpy as np
from .controller_base import Controller
import scipy.linalg as linalg

class Lqr(Controller):
    def __init__(self, a, b, p, x_sp=[], u_sp =[]):
        self.p_ = linalg.solve_discrete_are(a, b, p['q'], p['r'])
        self.k_ = -np.linalg.inv(b.T @ self.p_ @ b + p['r']) @ b.T @ self.p_ @ a
        self.x_sp_ = x_sp
        self.u_sp_ = u_sp
    def evaluate(self, x):
        if len(self.u_sp_) != 0 and len(self.x_sp_)!=0:
            return self.u_sp_ + self.k_ @ (x-self.x_sp_), None
        else:
            return self.k_ @ x, None