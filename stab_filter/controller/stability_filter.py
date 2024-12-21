'''
Copyright (c) 2023, ETH Zurich, Alexandre Didier, Andrea Zanelli, Kim P. Wabersich, Prof. Dr. Melanie N. Zeilinger, 
Institute for Dynamic Systems and Control, D-MAVT
'''

import casadi as ca
from scipy.optimize import linprog
import numpy as np
import polytope as pt
from .lqr import Lqr
import cvxpy as cp
    
class StabilityFilter():
    def support_function(self, a, ax, bx):
        if a.ndim > 1:
            supp = np.zeros(a.shape[0])
            for i in range(0, a.shape[0]):
                x_opt = linprog(-a[i,:], A_ub=ax, b_ub=bx)
                supp[i] = a[i,:] @ x_opt.x
            return supp
        else:
            x_opt = linprog(-a, A_ub=ax, b_ub=bx)
            return a @ x_opt.x
        
    def tube_and_terminal_computation(self, a, b, k, x_sp, u_sp, aw, bw, p):
        ax = p['Ax'][:, 1:]
        bx = p['bx'] - ax @ x_sp
        au = p['Au']
        bu = p['bu'] - au @ u_sp
        acl = a + b @ k
        # compute iterative constraint tightening
        nw = aw.shape[1]
        opti = ca.Opti()
        opti.solver('ipopt',{'ipopt.print_level':0, 'print_time':0})
        w_array = []
        self.ax = ax
        self.au = au
        self.bx_tightened = [bx]
        self.bu_tightened = [bu]
        err = np.zeros((nw,1))
        for i in range(1, p['n_mpc']):
            print(f'Compute constraint tightening prediction step {i}')
            bx_tight_i = np.zeros(ax.shape[0])
            bu_tight_i = np.zeros(au.shape[0])
            w_array.append(opti.variable(nw,1))
            opti.subject_to(aw @ w_array[-1] <= bw)
            err = acl @ err + w_array[-1]
            for ii in range(ax.shape[0]):
                obj = ax[ii,:].reshape((1, p['nx']-1)) @ err
                opti.minimize(-obj)
                sol = opti.solve()
                bx_tight_i[ii] = sol.value(obj)
            for ii in range(au.shape[0]):
                obj = au[ii,:].reshape((1, p['nu'])) @ k @ err
                opti.minimize(-obj)
                sol = opti.solve()
                bu_tight_i[ii] = sol.value(obj)
            self.bx_tightened.append(bx - bx_tight_i)
            self.bu_tightened.append(bu - bu_tight_i)
        # Compute maximum positive terminal invariant set
        convergence = False
        P_omega = pt.Polytope(ax, bx)
        while not convergence:
            pre_A_omega = P_omega.A @ acl
            pre_b_omega = P_omega.b.copy()
            pre_b_omega -= self.support_function(pre_A_omega, aw, bw)
            pre_A_intersect = np.vstack((P_omega.A, pre_A_omega))
            pre_b_intersect = np.hstack((P_omega.b, pre_b_omega))
            P_intersect = pt.Polytope(pre_A_intersect, pre_b_intersect)
            P_intersect = pt.reduce(P_intersect)
            if pt.is_subset(P_intersect, P_omega):
                V = pt.extreme(P_intersect)
                for v in V:
                  assert(np.max((P_omega.A @ v - P_omega.b).ravel()) <=0.0000001)
                convergence = True
            print(f'Maximum robust invariant set iteration')
            P_omega = P_intersect
        ax_term = P_omega.A
        bx_term = P_omega.b
        # Compute tightening of terminal positive invariant set
        w_array.append(opti.variable(nw,1))
        opti.subject_to(aw @ w_array[-1] <= bw)
        err = acl @ err + w_array[-1]
        bx_term_tight = np.zeros(ax_term.shape[0])
        for ii in range(ax_term.shape[0]):
            obj = ax_term[ii,:].reshape((1, p['nx']-1)) @ err
            opti.minimize(-obj)
            sol = opti.solve()
            bx_term_tight[ii] = sol.value(obj)
        self.ax_term = ax_term
        self.bx_term = bx_term
        self.bx_term_tightened = (bx_term - bx_term_tight) * 0.01 # make smaller to avoid switch in terminal set
    
    def __init__(self, a, b, x_sp, u_sp, aw, bw, p):
        # tube controller and terminal ingredient
        lqr = Lqr(a, b, p)
        k_lqr, p_lqr = lqr.k_, lqr.p_
        # determine constraint tightening
        self.tube_and_terminal_computation(a, b, k_lqr, x_sp, u_sp, aw, bw, p)
        # similar to mpc controller but with different constraint tightening

        nx = a.shape[0]
        nu = b.shape[1]
        x0 = cp.Parameter((nx,)) # current state
        x1pred = cp.Parameter((nx,)) # predicted state from last time step

        x_pred = cp.Variable((nx, p['n_mpc'])) # predicted states Deltax
        u_pred = cp.Variable((nu, p['n_mpc']-1)) # predicted inputs Deltau
        
        # dynamics
        constraints = []
        constraints += [x_pred[:,0] == x0] #initial constraint

        for i in range(p['n_mpc']-1):
            constraints += [x_pred[:,i+1] == a @ x_pred[:,i] + b @ u_pred[:,i]] #dynamics

        # state constraints
        for i in range(1, p['n_mpc']):
            constraints += [self.ax @ x_pred[:,i] <= self.bx_tightened[i]] # set point already included

        # input constraints
        for i in range(p['n_mpc']-1):
            constraints += [self.au @ u_pred[:,i] <= self.bu_tightened[i]] # set point already included

        # terminal constraint
        constraints += [self.ax_term @ x_pred[:,-1] <= self.bx_term_tightened]

        # store member functions
        self.nu_ = nu
        self.nx_ = nx
        self.a_ = a
        self.b_ = b
        self.k_lqr_ = k_lqr
        self.p_lqr_ = p_lqr
        self.x_pred_ = x_pred
        self.u_pred_ = u_pred
        self.x0_ = x0
        self.x1pred_ = x1pred
        self.x_sp_ = x_sp
        self.u_sp_ = u_sp

        # stability filter related changes from mpc
        
        ud = cp.Parameter((self.nu_,)) # desired input
        u0_ = cp.Parameter((self.nu_,)) # first warm-start input
        v = cp.Parameter((1,)) # constructed stability cost at each time step
        l = cp.Parameter((1,)) # stage cost

        # construct lyapunov function at next time step
        v_pred = 0
        for i in range(1, p['n_mpc']-1):
            v_pred += cp.quad_form(self.x_pred_[:,i], p['q']) + cp.quad_form(self.u_pred_[:,i], p['r'])

        u_pred_cand_last = self.k_lqr_ @ self.x_pred_[:,-1]
        v_pred += cp.quad_form(self.x_pred_[:,-1], p['q']) + cp.quad_form(u_pred_cand_last, p['r'])
        x_pred_cand_last = self.a_ @ self.x_pred_[:,-1] + self.b_ @ u_pred_cand_last
        v_pred += cp.quad_form(x_pred_cand_last.T, self.p_lqr_)

        constraints += [v_pred <= v - l ] 

        # filtering objective

        self.obj_ = cp.norm(ud - self.u_pred_[:,0], 2)**2
        self.prob = cp.Problem(cp.Minimize(self.obj_), constraints)

        # init candidate input/generator
        self.u_last_ = 1e1 * np.ones((self.nu_, p['n_mpc']))

        # store additional member functions
        self.v_ = v
        self.l_ = l
        self.u0_ = u0_
        self.ud_ = ud
        self.v_pred_ = v_pred
        self.n_mpc_ = p['n_mpc']
        self.q_ = p['q']
        self.r_ = p['r']
        self.rho_ = p['rho1']

    def evaluate(self, ud, x):
        self.ud_.value = (ud - self.u_sp_)
        self.x0_.value = x - self.x_sp_

        # compute stabilization constraint
        v = 0
        x_pred = x.copy() - self.x_sp_ # current state Delta x
        w = x_pred - self.x1pred_.value # incurred disturbance
        u_ws_ = np.zeros((self.nu_, self.n_mpc_-1)) # warm start input
        x_ws_ = np.zeros((self.nx_, self.n_mpc_)) # warm start state
        x_ws_[:,0] = x_pred 

        # check if in terminal set:
        max_terminal_half_space = np.max((self.ax_term @ x_pred - self.bx_term_tightened).ravel())
        
        # compute warmstart input and cost
        for i in range(self.n_mpc_-2):
            u = self.u_last_[:,i+1] + self.k_lqr_ @ np.linalg.matrix_power(self.a_ + self.b_ @ self.k_lqr_, i) @ w
            u_ws_[:,i] = u
            v += x_pred.T @ self.q_ @ x_pred + u.T @ self.r_ @ u
            x_pred = self.a_ @ x_pred + self.b_ @ u
            x_ws_[:,i+1] = x_pred
        u = self.k_lqr_ @ x_pred + self.k_lqr_ @ np.linalg.matrix_power(self.a_ + self.b_ @ self.k_lqr_, self.n_mpc_-1) @ w
        u_ws_[:,-1] = u
        v += x_pred.T @ self.q_ @ x_pred + u.T @ self.r_ @ u
        x_pred = self.a_ @ x_pred + self.b_ @ u
        x_ws_[:,-1] = x_pred
        v += x_pred.T @ self.p_lqr_ @ x_pred

        # compute lqr cost if in terminal set
        v_lqr=float('inf')
        if max_terminal_half_space <= 0:
          v_lqr = x_pred.T @ self.p_lqr_ @ x_pred

        if v <= v_lqr:
            u0_ = u_ws_[:,0]
        else:
            v = v_lqr
            u0_ = self.k_lqr_ @ (x.copy() - self.x_sp_)
            x_ws_[:,0] = x.copy() - self.x_sp_
            for i in range(self.n_mpc_-1):
                u_ws_[:,i] = self.k_lqr_ @ x_ws_[:,i]
                x_ws_[:,i+1] = self.a_ @ x_ws_[:,i] + self.b_ @ u_ws_[:,i]

        self.u0_.value = u0_
        self.v_.value = np.array([v])
        l = (1-self.rho_) * ((x.copy() - self.x_sp_).T @ self.q_  @ (x.copy() - self.x_sp_) + u0_.T @ self.r_ @ u0_)
        self.l_.value = np.array([l])

        try:
            self.prob.solve(verbose=False, solver=cp.MOSEK)
            if self.prob.status != cp.OPTIMAL:
                self.u_last_ = u_ws_
                self.x1pred_.value = x_ws_[:,1]
            else:
                self.u_last_ = self.u_pred_.value
                self.x1pred_.value = self.x_pred_[:,1].value
        except Exception as e:
            self.u_last_ = u_ws_
            self.x1pred_.value = x_ws_[:,1]
        

        u = self.u_last_[:, 0] + self.u_sp_

        # control information
        print(f'Steering={ud[0]:.2f} -> {u[0]:.2f} | Torque={ud[1]:.2f} -> {u[1]:.2f} | V(x,w) = {v:.2f}')
        ctrl_info = {'v':v, 'udev': np.linalg.norm(ud - u), 'w': w}
        return u, ctrl_info