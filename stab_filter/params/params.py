'''
Copyright (c) 2023, ETH Zurich, Alexandre Didier, Andrea Zanelli, Kim P. Wabersich, Prof. Dr. Melanie N. Zeilinger, 
Institute for Dynamic Systems and Control, D-MAVT
'''

import numpy as np
def CarParams():
    # @title Experiment request data
    p = dict(
    ### Simulation parameters
    vdes = 1.1, # target speed for simulation
    psi0 = np.deg2rad(12), #  E.g., due to aquaplaning, ice, etc, vx=0, vy=vdes, but yaw \neq 0
    # 15: Demonstrates that almost all proposed inputs can be passed through. Some will be filtered due to safety constraint.
    tsim = 5, # simulation time in seconds
    ### Predictive control parameters
    n_mpc = 70, # planning horiozn
    q = 2 * np.diag([1,1,1,1,1,1]),  # state cost for stabilization task
    r = 1 * np.diag([1,1]), # input cost for stabilization task
    q_driver = 0.0001 * np.diag([1,1,1,1,1,1]),  # LQR human model/"driver reacting too slow"
    r_driver = 1 * np.diag([1,1]), # LQR human model
    rho1 = 0.0, # for plotting
    rho2 = 0.99, # for plotting
    rho3 = 1000, # for plotting
    ### Kinematic bicycle model with pacejka tire model
    state_labels = ['x', 'y', 'psi', 'vx', 'vy', 'psid', 'delta'],
    input_labels = ['delta_d', 'tau'],
    dt = 0.01, #  discretization
    T_delta = 0.1, # steering pt1 approximation
    nx = 7,   # number of states
    nu = 2,   # number of inputs
    l_f = 0.051, # c.g. -> front axis
    l_r = 0.046, # c.g. -> rear axis
    I_zz = 0.000315, # yaw intertia
    m = 0.184, # mass
    fx = [1.085, 0.577, -0.2085,-0.000435, -0.0993], # drivetrain/braking parameters
    D_F = 0.75, # Pacejka param
    C_F = 1.5, # Pacejka param
    B_F = 4, # Pacejka param
    D_R = 1.05, # Pacejka param
    C_R = 1.45, # Pacejka param
    B_R = 6.5, # Pacejka param
    D_F_low = 0.31, # Pacejka param
    C_F_low = 1.29, # Pacejka param
    B_F_low = 26, # Pacejka param
    D_R_low = 0.3, # Pacejka param
    C_R_low = 1.29, # Pacejka param
    B_R_low = 55, # Pacejka param
    u_box = np.array([ # box input constraints
        [-np.deg2rad(35),  # minimum steering angle
         np.deg2rad(35)], # maximum steering angle
        [-2,     # minimum torque
         2],     # maximum torque
    ]),
    Ax = np.array([ # polytopic state constraints
        [0, 1, 0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0, 0],
        [0,  0, 1, 0, 0, 0, 0],
        [0,  0,-1, 0, 0, 0, 0],
        [0,  0, 0, 1, 0, 0, 0],
        [0,  0, 0,-1, 0, 0, 0],
        [0,  0, 0,0, 1, 0, 0],
        [0,  0, 0,0, -1, 0, 0],
        [0,  0, 0,0, 0, 1, 0],
        [0,  0, 0,0, 0, -1, 0],
        [0,  0, 0,0, 0, 0, 1],
        [0,  0, 0,0, 0, 0, -1]
    ]),
    bx = np.array([
        0.1, # maximum y off
        0.1, # -minimum y off
        np.deg2rad(15), # maximum psi
        np.deg2rad(15), # -minimum psi
        1.3, # maximum vx
        -0.1, # -minimum vx
        0.5, # maximum vy
        0.5,# -minimum vy
        np.pi/2, # maximum psid
        np.pi/2, # -minimum psid
        np.deg2rad(35), # maximum delta
        np.deg2rad(35) # -minimum delta
        ]),
    n_w_samples = 3000 # number of sample for approximating disturbance
    )
    return p