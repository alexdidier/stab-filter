'''
Copyright (c) 2023, ETH Zurich, Alexandre Didier, Andrea Zanelli, Kim P. Wabersich, Prof. Dr. Melanie N. Zeilinger, 
Institute for Dynamic Systems and Control, D-MAVT
'''

import casadi as ca
import numpy as np
import polytope as pt
from plotly.subplots import make_subplots

def sample_from_box(n_samples, box):
    n = len(box)
    samples = np.zeros((n_samples, n))
    for i in range(0, n_samples):
        z1 = np.random.rand(n)
        for ii in range(0, n):
            scale = box[ii][1] - box[ii][0]
            samples[i, ii] = box[ii][0] + scale * z1[ii]
    return samples

def sample_from_polytope(n, ax, bx):
    nx = ax.shape[1]
    samples = np.zeros((n, nx))
    n_succ = 0
    p = pt.Polytope(ax, bx)
    bbox = pt.bounding_box(p)
    bbox_ = []
    for i in range(0, nx):
        bbox_.append([bbox[0][i], bbox[1][i]])
    while n_succ < n:
        # sample new data
        samples_raw = sample_from_box(n, bbox_)
        # reject non-valid samples
        for i in range(0, n):
            if np.max(ax @ samples_raw[i,:].T - bx) <= 0 and n_succ < n:
                samples[n_succ,:] = samples_raw[i, :]
                n_succ+=1

    return samples

def box_clip(u, box):
    uc = np.zeros_like(u)
    for i in range(len(u)):
        uc[i] = max(u[i], box[i,0])
        uc[i] = min(uc[i], box[i,1])
    return uc

def sim(dyn, control_action, x0, p):
    nsim = int(p['tsim']/p['dt']) + 1
    xsim = np.zeros((p['nx'], nsim))
    usim = np.zeros((p['nu'], nsim))
    wsim = np.zeros((1, nsim))
    vsim = np.zeros((1, nsim))
    udevsim = np.zeros((1, nsim))
    xsim[:,0] = x0
    u, ctrl_info = control_action(x0)
    usim[:,0] = box_clip(u, p['u_box'])
    vsim[:,0] = ctrl_info['v'] if type(ctrl_info) == dict else 0
    udevsim[:,0] = ctrl_info['udev'] if type(ctrl_info) == dict else 0
    # simulation
    for k in range(1,nsim):
        xsim[:,k] = dyn(xsim[:,k-1], usim[:,k-1])
        u, ctrl_info = control_action(xsim[:,k])
        usim[:,k] = box_clip(u, p['u_box'])
        vsim[:,k] = ctrl_info['v'] if type(ctrl_info) == dict else 0
        udevsim[:,k] = ctrl_info['udev'] if type(ctrl_info) == dict else 0
        wsim[:,k] = ca.norm_2(ctrl_info['w']) if type(ctrl_info) == dict else 0
    return xsim,usim,vsim,udevsim, wsim

def sim_raw_to_dict(xsim, usim, p, vsim=[], udevsim = []):
    nsim = len(xsim[0,:])
    xsim = {
            **{"time":np.array([p['dt']*i for i in range(nsim)])},
            **{p['state_labels'][i]:xsim[i,:] for i in range(p['nx'])}
        }
    usim = {
            **{"time":np.array([p['dt']*i for i in range(nsim)])},
            **{p['input_labels'][i]:usim[i,:] for i in range(p['nu'])}
        }
    if len(vsim) != 0 and len(udevsim) != 0:
        vsim = {
                **{"time":np.array([p['dt']*i for i in range(nsim)])},
                **{"v":vsim.ravel()}
            }
        udevsim = {
                **{"time":np.array([p['dt']*i for i in range(nsim)])},
                **{"udev":udevsim.ravel()}
            }
        return xsim, usim, vsim, udevsim
    return xsim, usim

def dyn(x,u,p):
    """Miniature race car dynamics with normal friction"""
    px, py, psi, vx, vy, psid, delta = x[0],x[1],x[2],x[3],x[4],x[5],x[6]
    delta_d, tau = u[0],u[1]
    # Tire model
    alpha_F = ca.atan2( (-vy - p['l_f'] * psid) ,  vx ) + delta
    alpha_R = ca.atan2( (-vy + p['l_r'] * psid) ,  vx )
    F_fy = p['D_F'] * ca.sin(p['C_F'] * ca.atan(p['B_F'] * alpha_F))
    F_ry = p['D_R'] * ca.sin(p['C_R'] * ca.atan(p['B_R'] * alpha_R))
    # drive train/braking
    F_x = p['fx'][0]*tau + p['fx'][1]*tau**2 + p['fx'][2]*vx + p['fx'][3]*vx**2 + p['fx'][4]*vx*tau
    # continuous dynamics
    f = ca.vcat([
        vx * ca.cos(psi) - vy * ca.sin(psi),
        vx * ca.sin(psi) + vy * ca.cos(psi),
        psid,
        (1/p['m']) * (F_x - F_fy * ca.sin(delta) + p['m'] * vy * psid),
        (1/p['m']) * (F_ry + F_fy * ca.cos(delta) - p['m'] * vx * psid),
        (1/p['I_zz']) * (F_fy * p['l_f'] * ca.cos(delta) - F_ry * p['l_r']),
        delta + (1/p['T_delta']) * (delta_d - delta)
        ])
    return f

def dyn_low(x,u,p):
    """Miniature race car dynamics with low friction"""
    px, py, psi, vx, vy, psid, delta = x[0],x[1],x[2],x[3],x[4],x[5],x[6]
    delta_d, tau = u[0],u[1]
    # Tire model
    alpha_F = ca.atan2( (-vy - p['l_f'] * psid) ,  vx ) + delta
    alpha_R = ca.atan2( (-vy + p['l_r'] * psid) ,  vx )
    F_fy = 0.3 * p['D_F_low'] * ca.sin(p['C_F_low'] * ca.atan(p['B_F_low'] * alpha_F))
    F_ry = 0.3 * p['D_R_low'] * ca.sin(p['C_R_low'] * ca.atan(p['B_R_low'] * alpha_R))
    # drive train/braking
    F_x = p['fx'][0]*tau + p['fx'][1]*tau**2 + p['fx'][2]*vx + p['fx'][3]*vx**2 + p['fx'][4]*vx*tau
    # continuous dynamics
    f = ca.vcat([
        vx * ca.cos(psi) - vy * ca.sin(psi),
        vx * ca.sin(psi) + vy * ca.cos(psi),
        psid,
        (1/p['m']) * (F_x - F_fy * ca.sin(delta) + p['m'] * vy * psid),
        (1/p['m']) * (F_ry + F_fy * ca.cos(delta) - p['m'] * vx * psid),
        (1/p['I_zz']) * (F_fy * p['l_f'] * ca.cos(delta) - F_ry * p['l_r']),
        delta + (1/p['T_delta']) * (delta_d - delta)
        ])
    return f

def discretize(f, x, u, p):
    """Runge-Kutta 4"""
    k1 = f(x,u, p)
    k2 = f(x + p['dt'] * 0.5 * k1, u, p)
    k3 = f(x + p['dt'] * 0.5 * k2, u, p)
    k4 = f(x + p['dt'] * k3, u, p)
    x_next = x + (1/6) * p['dt'] * (k1 + 2 * k2 + 2 * k3 + k4)
    return x_next.T

def compute_steady_state(f, delta_d, vx_d, p):
    """Computes steady state based on applied steering angle delta_d by driver, given current velocity"""
    # compute desired yaw rate
    opti_s = ca.Opti()
    xsp = opti_s.variable(p['nx'],1)
    usp = opti_s.variable(p['nu'],1)
    xnext = f(xsp,usp)
    # steady-state conditions based on applied input and current vel.
    opti_s.subject_to(usp[0] == delta_d) # steering
    opti_s.subject_to(xsp[3] == vx_d) # velocity
    # steady-state equation
    opti_s.subject_to(xnext[3] == xsp[3]) # longitudinal velocity
    opti_s.subject_to(xnext[4] == xsp[4]) # lateral velocity
    opti_s.subject_to(xnext[5] == xsp[5]) # yaw-rate
    opti_s.subject_to(xnext[6] == xsp[6]) # steering angle
    # enforce constraints
    opti_s.subject_to(p['Ax']@xsp <= p['bx'])
    opti_s.subject_to(p['u_box'][:,0].T <= usp)
    opti_s.subject_to(usp <= p['u_box'][:,1])
    opti_s.minimize(xsp.T @ xsp) # if multiple solutions exist
    # solve problem
    opti_s.set_initial(xsp,np.array([0,0,0,vx_d,0,0,0]))
    opti_s.solver('ipopt',{'ipopt.print_level':0, 'print_time':0})
    sol = opti_s.solve()
    return sol.value(xsp), sol.value(usp)

def linearize_dyn(dyn, xsp, usp):
    n = len(xsp)
    m = len(usp)
    x = ca.MX.sym('x', n, 1)
    u = ca.MX.sym('u', m, 1)
    x_val = ca.MX.sym('x', n, 1)
    u_val = ca.MX.sym('u', m, 1)
    x_plus = dyn(x, u)
    j_f_x = ca.jacobian(x_plus, x)
    j_f_x_eval = ca.Function('j_f_x', [x_val, u_val], ca.substitute(
        [j_f_x], [x, u], [x_val, u_val]))
    j_f_u = ca.jacobian(x_plus, u)
    j_f_u_eval = ca.Function('j_f_u', [x_val, u_val], ca.substitute(
        [j_f_u], [x, u], [x_val, u_val]))
    a = np.array(j_f_x_eval(xsp, usp))
    b = np.array(j_f_u_eval(xsp, usp))
    return a,b

def get_disturbance_set(p, x_sp, u_sp, dyn_d_low, a_low, b_low):
    x_samples = sample_from_polytope(p['n_w_samples'], p['Ax'][:,2:], p['bx'])
    u_samples = sample_from_box(p['n_w_samples'], p['u_box'])
    wbox = np.zeros((p['nx']-1,2))
    for i in range(p['n_w_samples']):
        xp_nonlin = dyn_d_low(np.hstack((0,0,x_samples[i,:])).T, u_samples[i,:].T)
        xp_lin = a_low @ (np.hstack((0,x_samples[i,:])).T - x_sp[1:]) + b_low @ (u_samples[i,:].T-u_sp) + x_sp[1:]
        error = np.array(xp_nonlin[1:]).ravel() - xp_lin.ravel()
        wbox[:,0] = np.minimum(wbox[:,0], error)
        wbox[:,1] = np.maximum(wbox[:,1], error)
    wbox = 0.03 * wbox # counteract conservative hyperbox bounding
    P_w = pt.box2poly(wbox)
    wbox_full = np.vstack((np.zeros(2),wbox))
    return wbox_full, P_w