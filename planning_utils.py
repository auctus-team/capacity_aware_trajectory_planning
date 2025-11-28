import qpsolvers
import numpy as np
from spatialmath import SE3,SO3, base
import spatialmath
import pinocchio as pin
import time

from cvxopt import matrix
import cvxopt.glpk

from scipy.optimize import linprog
from scipy.linalg import block_diag

from pynocchio import RobotWrapper

def solve_lp(c, A_eq, b_eq, x_min, x_max):
    # scipy linprog
    # res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=np.array([x_min,x_max]).T)
    # return res.x
    
    # glpk linprog
    c = matrix(c)
    A = matrix(A_eq)
    b = matrix(b_eq)
    G = matrix(np.vstack((-np.identity(len(x_min)),np.identity(len(x_min)))))
    h = matrix(np.hstack((list(-np.array(x_min)),x_max)))
    solvers_opt={'tm_lim': 100000, 'msg_lev': 'GLP_MSG_OFF', 'it_lim':10000}
    res = cvxopt.glpk.lp(c=c,  A=A, b=b, G=G,h=h, options=solvers_opt)
    return np.array(res[1]).reshape((-1,))


def solve_qp(A,s,x_min,x_max, grad = None, reg_w=1e-7, solver=None):
    
    if (solver is None) or('cvxopt' in solver ) :
        A =np.matrix(A)
        s = np.matrix(s)

        P = A.T@A
        if grad is not None:
            grad = np.matrix(grad).reshape(1,-1)
            q = matrix(-A.T@s + reg_w*grad.T)
            P = P + np.eye(len(grad))*reg_w
        else:
            q = matrix(-A.T@s)
        P = matrix(P)
        G = matrix(np.vstack((-np.identity(len(x_max)),np.identity(len(x_min)))))
        h = matrix(np.hstack((list(-np.array(x_min)),x_max)))
        return np.array(cvxopt.solvers.qp(P, q, G, h)['x']).flatten()
    
    else:

        P = A.T@A
        if grad is not None:
            q = -A.T@s + reg_w*grad[:,None]
            P= P+np.eye(len(grad))*reg_w
        else:
            q = -A.T@s
        G = np.vstack((-np.identity(len(x_max)),np.identity(len(x_min))))
        h = np.hstack((list(-np.array(x_min)),x_max))

        return qpsolvers.solve_qp(P, q.flatten(), G,h, solver=solver)
    
from copy import copy
from pathlib import Path
from sys import path
import time

from ruckig import InputParameter, OutputParameter, Result, Ruckig

class TrajData():
    name = ''
    
    
def compute_capacity_aware_trajectory(X_init, X_final,q0,  robot=None, robot_pyn=None, lims=[], scale=1.0, options = None , verbose=False):
    
    
    # check the type of the X_init and X_final
    if type(X_init) == spatialmath.pose3d.SE3:
        X_init = pin.SE3(X_init.R, X_init.t)
        X_final = pin.SE3(X_final.R, X_final.t)
    elif type(X_init) == np.ndarray:
        X_init = pin.SE3(X_init[:3,:3], X_init[:3,3])
        X_final = pin.SE3(X_final[:3,:3], X_final[:3,3])
    elif type(X_init) == pin.SE3:
        pass
    else:
        print("Unknown type of X_init")
        return
    
    if robot_pyn is not None:
        X_r = robot_pyn.forward(q0)
        n_dof = robot_pyn.n
    elif robot is not None:
        X_r = robot.fkine(q0)
        n_dof = robot.n
    else:
        print("Please provide either robot or robot_pyn")
        return


    if options is not None and 'uptate_current_position' in options.keys():
        uptate_current_position = options['uptate_current_position']
    else: 
        uptate_current_position = False
    
    if options is not None and 'qp_form' in options.keys():
        qp_form = options['qp_form']
    else: 
        qp_form = 'acceleration'
    
    
    if options is not None and 'calculate_limits' in options.keys():
        calculate_limits = options['calculate_limits']
    else: 
        calculate_limits = True
        
    if options is not None and 'scale_limits' in options.keys():
        scale_limits = options['scale_limits']
    else: 
        scale_limits = True

    if options is not None and 'stop_on_singular' in options.keys():
        stop_on_singular = options['stop_on_singular']
    else: 
        stop_on_singular = False
    
    if options is not None and 'downsampling_ratio' in options.keys():
        n_ruckig = options['downsampling_ratio']
    else:
        n_ruckig = 1.0

    if options is not None and 'jerk_scale' in options.keys():
        jerk_scale = options['jerk_scale']
    else:
        jerk_scale =scale
    
    if lims is None:
        print('no limits specified')
        data = []
        return
    else:
        if scale_limits:
            dq_max = scale*lims['dq_max'].copy()
            ddq_max = scale*lims['ddq_max'].copy()
            dddq_max = jerk_scale*lims['dddq_max'].copy()
            t_max = scale*lims['t_max'].copy()
        else:
            dq_max = lims['dq_max'].copy()
            ddq_max = lims['ddq_max'].copy()
            dddq_max = lims['dddq_max'].copy()
            t_max = lims['t_max'].copy()
        q_min, q_max = lims['q_min'].copy(), lims['q_max'].copy()
        dq_min = -dq_max
        ddq_min = -ddq_max
        dddq_min = -dddq_max
        t_min = -t_max
        
        # cartesian space
        dddx_max = scale*lims['dddx_max'].copy()
        dddx_min = -dddx_max
        ddx_max = scale*lims['ddx_max'].copy()
        ddx_min = -ddx_max
        dx_max = scale*lims['dx_max'].copy()
        dx_min = -dx_max   
    
    if options is not None and 'scaled_qp_limits' in options.keys():
        scaled_qp_limits = options['scaled_qp_limits']
    else:
        scaled_qp_limits = True
        
    if options is not None and 'Kp' in options.keys():
        Kp = options['Kp']
        Kd = options['Kd']
        Ki = options['Ki']
    else:
        Kp = 170
        Kd = 40       
        Ki = 0
        
    if options is not None and 'clamp_velocity' in options.keys():
        clamp_velocity = options['clamp_velocity']
    else: 
        clamp_velocity = True
        
    if options is not None and 'clamp_min_accel' in options.keys():
        clamp_min_accel = options['clamp_min_accel']
    else: 
        clamp_min_accel = False
        
        
        
    if options is not None and 'override_acceleration' in options.keys():
        override_acceleration = options['override_acceleration']
    else: 
        override_acceleration = True
    

    if options is not None and 'use_manip_grad' in options.keys():
        use_manip_grad = options['use_manip_grad']
    else:
        use_manip_grad = False
    
    if use_manip_grad and robot is None:
        print("We need RTB model to calculate the manipulability gradient")
        return

    if options is not None and 'manip_grad_w' in options.keys():
        manip_grad_w = options['manip_grad_w']
    else:
        manip_grad_w =  50000.0

    data = TrajData()
    u = np.zeros(6)
    u[:3] = X_final.translation-X_init.translation
    d = np.linalg.norm(u)
    u = u/d

    U,S,V = np.linalg.svd(u[:,None])
    V2 = U[:,1:].T

    U,S,V = np.linalg.svd(u[:3,None])
    V23 = U[:,1:].T
    
    if options is not None and 'dt' in options.keys():
        dt = options['dt']
    else:
        dt =  0.001
    # n_ruckig = 10.0
    dt_ruckig = n_ruckig*dt

    
    if options is not None and 'Tf' in options.keys():
        Tf = options['Tf']
        alpha = Tf/(Tf + dt)
    else:
        Tf = 0.00
        alpha = Tf/(Tf + dt)

    # Create instances: the Ruckig OTG as well as input and output parameters
    otg = Ruckig(1, dt_ruckig)  # DoFs, control cycle
    inp = InputParameter(1)
    out = OutputParameter(1)

    # Set input parameters
    inp.current_position = [0.0]
    inp.current_velocity = [0.0]
    inp.current_acceleration = [0.0]

    inp.target_position = [d]
    inp.target_velocity = [0.0]
    inp.target_acceleration = [0.0]

    inp.max_velocity = [dx_max[0]]
    inp.max_acceleration = [ddx_max[0]]
    inp.max_jerk = [dddx_max[0]]

    print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))

    # Generate the trajectory within the control loop
    data.qr_list = []
    data.x_list, data.dx_list, data.ddx_list, data.dddx_list = [], [], [], []
    data.x_q_list, data.dx_q_list, data.ddx_q_list, data.dddx_q_list = [], [], [], []
    data.dx_max_list, data.ddx_max_list, data.dddx_max_list = [], [], []
    data.dx_min_list, data.ddx_min_list, data.dddx_min_list = [], [], []
    data.e_pos_list_ruckig, data.e_rot_list_ruckig = [], []
    data.x3d_ruckig = []
    
    out_list = []
    res = Result.Working

    s = time.time()

    if robot_pyn is None:
        sol = robot.ikine_LM(X_final.np, q0)         # solve IK
        q_final = sol.q
        J = robot.jacob0(q_final)
    else:
        q_final = robot_pyn.ik(X_final, q0, qlim=True, verbose=False)
        J = robot_pyn.jacobian(q_final)
    c = u@J
    Aeq = V2@J
    beq = np.zeros(5)
    c_ext = np.hstack((c,c,c,-c))
    Aeq_ext=block_diag(Aeq,Aeq,Aeq,Aeq)
    beq_ext = b_eq=np.hstack((beq,beq,beq,beq))
    x_min_ext = np.hstack((dq_min,ddq_min,dddq_min,ddq_min))
    x_max_ext = np.hstack((dq_max,ddq_max,dddq_max,ddq_max))
    q_ext = solve_lp(-c_ext, Aeq_ext, beq_ext, x_min_ext, x_max_ext)
    ds_final = (c@q_ext[:n_dof])
    dds_final = (c@q_ext[n_dof:(2*n_dof)])
    ddds_final = (c@q_ext[(2*n_dof):(3*n_dof)])
    dds_min_final = (c@q_ext[(3*n_dof):(4*n_dof)])


    # bias compensaiton
    db_old, dds_min_old, dds_max_old = 0, 0, 0
    alpha_ds = 1.0
     
    err_sum = 0
        
    data.solved= True
    
    ruckig_current_accel, old_accel = [0], 0
    
    
    if robot_pyn is None:
        sol = robot.ikine_LM(X_init.np, q0)         # solve IK
        q_c = sol.q
    else:
        q_c = robot_pyn.ik(X_init, q0, qlim=True, verbose=False)
    
    dq_c, ddq_c = np.zeros(7), np.zeros(7)
    dddq_c = np.zeros(7)
    t_sim = 0
    t_sim_ruckig, t_old_ruckig = 0, -1
    out_position = 0.0
    out_velocity = 0.0
    out_acceleration = 0.0
    beq_filt = np.zeros(5)
    t0 = time.time()
    while res == Result.Working :# or data.dx_q_list[-1] > 1e-2:

        if t_sim > 10000: # more than 10sec
            print("ERROR: 10 sec simulation time exceeded! Stopping!")
            data.solved = False
            break
        
        if robot_pyn is None:
            J = robot.jacob0(q_c)
            X_c = robot.fkine(q_c)
            X_c = pin.SE3(X_c.R, X_c.t)
            J_dot = robot.jacob0_dot(q_c, dq_c)
        else:
            J = robot_pyn.jacobian(q_c)
            J_dot = robot_pyn.jacobian_dot(q_c, dq_c)
            X_c = robot_pyn.forward(q_c)


        # u[:3] = X_final.translation-X_c.translation
        # d = np.linalg.norm(u)
        # u = u/d
    
        # U,S,V = np.linalg.svd(u[:,None])
        # V2 = U[:,1:].T
    
        # U,S,V = np.linalg.svd(u[:3,None])
        # V23 = U[:,1:].T
        
        data.qr_list.append(q_c)
        data.x_q_list.append(u[:3]@(X_c.translation-X_init.translation))  
        data.x3d_ruckig.append(X_c.translation)
        data.dx_q_list.append(u@J@dq_c)     
        data.ddx_q_list.append(u@J@ddq_c + u@J_dot@dq_c)   
        data.dddx_q_list.append(u@J@dddq_c + u@J_dot@ddq_c) 


        # data.e_pos_list_ruckig.append(np.linalg.norm(V23@(X_c.translation-X_init.translation)))
        # data.e_rot_list_ruckig.append(np.linalg.norm(pin.log3(X_init.R.T@X_c.R)))


        if t_old_ruckig != t_sim_ruckig:
            if len(data.x_list) == 0:
                delta_x = (inp.current_position[0] )/dt_ruckig
                delta_dx = (inp.current_velocity[0])/dt_ruckig
                delta_ddx = (inp.current_acceleration[0])/dt_ruckig

                data.x_list.append([delta_x*dt])#+delta_dx*dt**2/2+delta_ddx*dt**2/6])
                data.dx_list.append([delta_dx*dt])#+delta_ddx*dt**2/2])
                data.ddx_list.append([delta_ddx*dt])
                data.dddx_list.append([delta_ddx])
            else:

                delta_x = (inp.current_position[0] - data.x_list[-1][0])/dt_ruckig
                delta_dx = (inp.current_velocity[0] - data.dx_list[-1][0])/dt_ruckig
                delta_ddx = (inp.current_acceleration[0] - data.ddx_list[-1][0])/dt_ruckig
                data.x_list.append([data.x_list[-1][0] + delta_x*dt])
                data.dx_list.append([data.dx_list[-1][0]+ delta_dx*dt])
                data.ddx_list.append([data.ddx_list[-1][0] + delta_ddx*dt])
                data.dddx_list.append([delta_ddx])

            out_position = inp.current_position[0]
            out_velocity = inp.current_velocity[0]
            out_acceleration = inp.current_acceleration[0]

            t_old_ruckig = t_sim_ruckig
        else:

            data.x_list.append([data.x_list[-1][0] + delta_x*dt])
            data.dx_list.append([data.dx_list[-1][0] + delta_dx*dt])
            data.ddx_list.append([data.ddx_list[-1][0] + delta_ddx*dt])  
            data.dddx_list.append([delta_ddx])

        if t_sim % n_ruckig == 0:
            if ds_final > 1e-1: ds_p=[ds_final]
            else: ds_p= []
            if dds_final > 1e-1: dds_p=[dds_final]
            else: dds_p= []
            if ddds_final > 1e-1: ddds_p=[ddds_final]
            else: ddds_p= []
            if dds_min_final < -1e-1: dds_min_p=[dds_min_final]
            else: dds_min_p= []

            # print(ds_p,dds_p,ddds_p)
            c = u@J
            Aeq = V2@J
            try:
                c_ext = np.hstack((c,c,c,-c))
                Aeq_ext=block_diag(Aeq,Aeq,Aeq,Aeq)
                
                #beq_filt = alpha*beq_filt  + (1 - alpha)*-V2@J_dot@dq_c
                beq_filt = -V2@J_dot@dq_c

                # alpha_ds = 1.0
                # db = u@J_dot@dq_c
                # ddb = (db - db_old) # gradient of the bias term
                # dds_min = c@dddq_min
                # if ddb:
                #     dt_dds_min_to_zero = dds_min/ddb
                # else:
                #     dt_dds_min_to_zero = 1.0
                # if dt_dds_min_to_zero < 0 and dt_dds_min_to_zero >- 1:
                #     alpha_ds = 1.0 - np.abs(dt_dds_min_to_zero)
                # else:
                #     alpha_ds = 1.0
                #     print(alpha_ds)
                # db_old = u@J_dot@dq_c

                # dds_max = c@dddq_max
                # if ddb:
                #     dt_dds_max_to_zero = dds_p[-1]/np.abs(ddb)
                # else:
                #     dt_dds_max_to_zero = 1.0
                # if ddb < 0 and dt_dds_max_to_zero > 0:# and dt_dds_max_to_zero < 10:
                #     alpha_ds = np.min([alpha_ds, (dt_dds_max_to_zero)/1000.0])
                #     print(alpha_ds, dt_dds_max_to_zero)
                # else:
                #     alpha_ds = np.min([alpha_ds, 1.0])
                
                beq_ext = np.hstack((beq,-V2@J_dot@dq_c,-V2@J_dot@ddq_c,beq_filt))
                #beq_ext = np.hstack((beq,beq,beq,beq))
                x_min_ext = np.hstack((alpha_ds*dq_min,ddq_min,dddq_min,ddq_min))
                x_max_ext = np.hstack((alpha_ds*dq_max,ddq_max,dddq_max,ddq_max))
                q_ext = solve_lp(-c_ext, Aeq_ext, beq_ext, x_min_ext, x_max_ext)
                ds_p.append(c@q_ext[:n_dof])
                dds_p.append(c@q_ext[n_dof:(2*n_dof)] + u@J_dot@dq_c)
                ddds_p.append(c@q_ext[(2*n_dof):(3*n_dof)] + u@J_dot@ddq_c)
                dds_min_p.append(c@q_ext[(3*n_dof):(4*n_dof)] + u@J_dot@dq_c)


                dt_to_zero = 1.0
                
                # dds_min_grad = dds_min_p[-1] - dds_min_old
                # if dds_min_grad :
                #     dt_dds_min_to_zero = np.abs(dds_p[-1])/np.abs(dds_min_grad)
                # else:
                #     dt_dds_min_to_zero = 10000.0
                # if dds_min_grad > 0 and dt_dds_min_to_zero > 0 and dt_dds_min_to_zero < dt_to_zero:
                #     alpha_ds = np.min([alpha_ds, 1-(dt_dds_min_to_zero)/dt_to_zero])
                # else:
                #     alpha_ds = np.min([alpha_ds, 1.0])
                    
                # dds_max_grad = (dds_p[-1] - dds_max_old)/0.001*dt_to_zero
                # # alpha_ds_new = 1.0
                # if dds_max_grad :
                #     dt_dds_max_to_zero = dds_p[-1]/np.abs(dds_max_grad)
                # else:
                #     dt_dds_max_to_zero = 10000.0
                # if dds_p[-1] + dds_max_grad < 0 or dds_p[-1] < 0:    
                #     # alpha_ds_new = np.min([alpha_ds_new, np.max([0.01, 0.1])])
                #     alpha_ds = alpha_ds-0.05
                # else:
                #     # alpha_ds_new = np.min([alpha_ds_new, 1.0])
                #     alpha_ds = alpha_ds+0.005
                
                # # alpha = 0.5
                # # alpha_ds = (1-alpha)* alpha_ds + alpha*alpha_ds_new          
                # alpha_ds = np.clip(alpha_ds, 0.01, 1)
                
            except:
                if verbose: print("except dds")
                # return
                beq = np.zeros(5)
                # ds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=np.zeros(5), x_min=dq_min, x_max=dq_max))
                # dds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=np.zeros(5), x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                # ddds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=np.zeros(5), x_min=dddq_min, x_max=dddq_max) + u@J_dot@ddq_c)
                # dds_min_p.append(c@solve_lp(c, A_eq=Aeq, b_eq=np.zeros(5), x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                
                # ds_p.append(ds_p[-1])
                # dds_p.append(dds_p[-1])
                # ddds_p.append(ddds_p[-1])
                # dds_min_p.append(dds_min_p[-1])

                # beq_filt = alpha*beq_filt  + (1 - alpha)*-V2@J_dot@dq_c
                beq_filt = -V2@J_dot@dq_c
                try:
                    ds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=beq, x_min=dq_min, x_max=dq_max))
                except:
                    ds_p.append(ds_p[-1])
                try:
                    dds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=-V2@J_dot@dq_c, x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                except:
                    done = False
                    for i in range(1,10):
                        try:
                            dds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=-V2@J_dot@dq_c/i, x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                            done = True   
                            break 
                        except:
                            continue
                    if not done:
                        dds_p.append(dds_p[-1])
                try:
                   ddds_p.append(c@solve_lp(-c, A_eq=Aeq, b_eq=-V2@J_dot@ddq_c, x_min=dddq_min, x_max=dddq_max) + u@J_dot@ddq_c)
                except:
                    ddds_p.append(ddds_p[-1])
                try:
                    dds_min_p.append(c@solve_lp(c, A_eq=Aeq, b_eq=beq_filt, x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                except:
                    done = False
                    for i in range(1,10):
                        try:
                            dds_min_p.append(c@solve_lp(c, A_eq=Aeq, b_eq=beq_filt/i, x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                            done = True   
                            break
                        except:
                            continue
                    if not done:
                        dds_min_p.append(dds_min_p[-1])

                
        # if len(data.dx_max_list):
        #     ds_p[-1] = 0.85*data.dx_max_list[-1] + 0.15*ds_p[-1]
        # if len(data.ddx_max_list) :
        #     dds_p[-1] = 0.85*data.ddx_max_list[-1] + 0.15*dds_p[-1]
        # if len(data.ddx_min_list):
        #     dds_min_p[-1] = 0.85*data.ddx_min_list[-1] + 0.15*dds_min_p[-1]
            
        data.dx_max_list.append(ds_p[-1])
        data.ddx_max_list.append(dds_p[-1])
        data.dddx_max_list.append(ddds_p[-1])
        data.dx_min_list.append(-ds_p[-1])
        data.ddx_min_list.append(dds_min_p[-1])
        data.dddx_min_list.append(-ddds_p[-1])

        if ds_p[-1]<=1e-4 :
            if verbose: print("Error - sinularity or infeasible position for ds")
            data = None 
            return
        if dds_p[-1]<=1e-4: 
            if verbose: print("Error - sinularity or infeasible position for dds")
            if stop_on_singular:
                data = None 
                return
            else:
                dds_p[-1] = 0.05
        if ddds_p[-1]<=1e-4:
            if verbose: print("Error - sinularity or infeasible position for ddds")
            if stop_on_singular:
                data = None 
                return
            else:
                ddds_p[-1] = 0.05
                
        if dds_min_p[-1]>=-1e-4: 
            if verbose: print("Error - sinularity or infeasible position for dds_min")
            if stop_on_singular:
                data = None 
                return
            else:
                dds_min_p[-1] = -0.05


        if uptate_current_position:
            # if current position is less than 2cm from the target one
            # update the current position
            # else dont update
            if np.abs(data.x_q_list[-1] - inp.target_position[0]) > 0.01:
                inp.current_position = [data.x_q_list[-1]]
            
        if t_sim % n_ruckig == 0:
        
                     
            tmp_max_vel =  inp.max_velocity[0]
            tmp_current_vel =  inp.current_velocity[0]
            
            if calculate_limits:
                if override_acceleration:
                    inp.current_acceleration = ruckig_current_accel
            
            clamped = False
            if calculate_limits:
                inp.max_velocity =[(ds_p[-1])]
                inp.max_acceleration=[(dds_p[-1])]
                inp.max_jerk=[(ddds_p[-1])]
                if clamp_min_accel:
                    inp.min_acceleration=[np.max(dds_min_p)]
                else:
                    inp.min_acceleration=[(dds_min_p[-1])]
            
                if clamp_velocity:
                    if inp.current_velocity[0] > inp.max_velocity[0]:
                        clamped = True
                        inp.current_velocity = inp.max_velocity

            res = otg.update(inp, out)
            out.pass_to_input(inp)
            # print(inp.current_acceleration)
        
            if calculate_limits:
                if clamped:
                    inp.current_acceleration = [np.max([(inp.max_velocity[0] - tmp_max_vel)/dt_ruckig, inp.min_acceleration[0]])]
                    
                
                if override_acceleration:
                    ruckig_current_accel = inp.current_acceleration
                    inp.current_acceleration = [(inp.current_velocity[0] - tmp_current_vel)/dt_ruckig]

            inp.current_acceleration = [(1-alpha)*inp.current_acceleration[0] + alpha*old_accel]
            old_accel = inp.current_acceleration[0]
            t_sim_ruckig = t_sim_ruckig + out.time;


        if 'acceleration' in qp_form:
            t_h = 0.01
            if not scaled_qp_limits:
                ddq_ub = np.vstack(
                    [ 
                        lims['ddq_max'],
                        (t_h*lims['dddq_max'] + ddq_c).flatten(), 
                        ((lims['dq_max'] - dq_c)/t_h).flatten(), 
                        (2*(lims['q_max'] - dq_c*t_h - q_c)/t_h**2).flatten()
                    ]).min(axis=0)
                ddq_lb = np.vstack(
                    [ 
                        -lims['ddq_max'], 
                        (t_h*-lims['dddq_max'] + ddq_c).flatten(), 
                        ((-lims['dq_max'] - dq_c)/t_h).flatten(), 
                        (2*(lims['q_min'] - dq_c*t_h - q_c)/t_h**2).flatten()
                    ]).max(axis=0)
            else:
                ddq_ub = np.vstack(
                    [ 
                        ddq_max, 
                        (t_h*dddq_max + ddq_c).flatten(), 
                        ((dq_max - dq_c)/t_h).flatten(), 
                        (2*(q_max - dq_c*t_h - q_c)/t_h**2).flatten()
                    ]).min(axis=0)
                ddq_lb = np.vstack(
                    [ 
                        ddq_min, 
                        (t_h*dddq_min + ddq_c).flatten(), 
                        ((dq_min - dq_c)/t_h).flatten(), 
                        (2*(q_min - dq_c*t_h - q_c)/t_h**2).flatten()
                    ]).max(axis=0)
        elif 'velocity' in qp_form:
            t_h = dt
            if not scaled_qp_limits:
                dq_ub = np.vstack(
                    [ 
                        lims['dq_max'], 
                        (t_h*lims['ddq_max'] + dq_c).flatten(), 
                        ((lims['q_max'] - q_c)/t_h).flatten(), 
                        (0.5*((t_h*5)**2)*lims['dddq_max'] + t_h*ddq_c + dq_c).flatten()
                    ]
                ).min(axis=0)
                dq_lb = np.vstack(
                    [
                        -lims['dq_max'], 
                        (t_h*-lims['ddq_max'] + dq_c).flatten(), 
                        ((lims['q_min'] - q_c)/t_h).flatten(), 
                        ((0.5*((t_h*5)**2)*-lims['dddq_max'] + t_h*ddq_c + dq_c).flatten())
                    ]
                 ).max(axis=0)
            else:
                dq_ub = np.vstack(
                    [ 
                        dq_max, 
                        (t_h*ddq_max + dq_c).flatten(), 
                        ((q_max - q_c)/t_h).flatten(), 
                        (0.5*((t_h*5)**2)*dddq_max + t_h*ddq_c + dq_c).flatten()
                    ]
                ).min(axis=0)
                dq_lb = np.vstack(
                    [
                        dq_min, 
                        (t_h*ddq_min + dq_c).flatten(), 
                        ((q_min - q_c)/t_h).flatten(), 
                        ((0.5*((t_h*5)**2)*dddq_min + t_h*ddq_c + dq_c).flatten())
                    ]
                 ).max(axis=0)
        # print('sc',dq_ub, dq_lb)
        # J = robot.jacob0(q_c)
        c = u@J
        q_c_old = q_c.copy()
        dq_c_old = dq_c.copy()
        ddq_c_old = ddq_c.copy()

        if 'acceleration' in qp_form:
            # acceleration feed-forward + pd
            ddx_des = data.ddx_list[-1]*u[:,None]
            ddx_des = ddx_des + Kd*(data.dx_list[-1]*u - J@dq_c_old)[:,None]
            # # only translation
            # # ddx_des[:3] = ddx_des[:3] + 70*SE3((np.array(data.x_list[-1]*u[:3]) - (robot.fkine(q_c_old).t - X_init.translation)).log(True)[:3,None]
            # translation + rotation
            X_dk = pin.SE3(X_init.rotation, X_init.translation+np.array(data.x_list[-1]*u[:3]))
            X_wa = pin.SE3(X_init.rotation, np.zeros(3))
            X_rk = pin.SE3(X_c.rotation, X_c.translation)
            X_log = X_dk.actInv(X_rk)
            log_dk = pin.log6(X_log)
            err = (-(X_wa.toActionMatrix()@log_dk)[:,None])
            err_sum = err_sum + err*dt
            ddx_des =  ddx_des + Kp*err + Ki*err_sum
            ddx_des = ddx_des - (J_dot@dq_c_old)[:,None]
            grad = (q0-q_c_old) + 2*(-dq_c_old)
            if use_manip_grad:
                grad = manip_grad_w*robot.jacobm(q_c).reshape((-1,))
    
            ddq_c = solve_qp(J,ddx_des, ddq_lb, ddq_ub, grad=-grad, reg_w=5*0.00001, solver='cvxopt')
            if ddq_c is None:
                print("No QP solution found")
                data.solved= False
                break
    
            dq_c = dq_c + ddq_c*dt
            q_c = np.clip(dq_c*dt + ddq_c*(dt**2)/2 + q_c, q_min,q_max)
            dq_c = np.clip(dq_c, dq_min, dq_max)
            dddq_c = (ddq_c - ddq_c_old)/dt

        elif 'velocity' in qp_form:
            # translation + rotation
            dx_des = data.dx_list[-1]*u[:,None]
            X_dk = pin.SE3(X_init.rotation, X_init.translation+np.array(data.x_list[-1]*u[:3]))
            X_rk = pin.SE3(X_c.rotation, X_c.translation)
            X_log = X_dk.actInv(X_rk)
            log_dk = pin.log6(X_log)
            dx_des = dx_des + Kp/1000.0*(-(X_dk.toActionMatrix()@log_dk)[:,None])
            
            grad = ((q_min+q_max)/2-q_c_old) + 0.1*(-dq_c_old)
            if use_manip_grad:
                grad = manip_grad_w*robot.jacobm(q_c).reshape((-1,))

            dq_c = solve_qp(J, dx_des, dq_lb, dq_ub, grad=-grad, reg_w=5*0.00001, solver='quadprog')
            if dq_c is None:
                print("No QP solution found")
                data.solved= False
                break
            
            
            # robot simulation
            if np.any(dq_c > dq_max) or np.any(dq_c < dq_min) : 
                print("vel outside limits")
            ddq_c = (dq_c - dq_c_old)/dt
            if np.any(ddq_c > 1.01*ddq_max) or np.any(ddq_c < 1.01*ddq_min) : 
                #print(ddq_c, ddq_max, ddq_min, ddq_c > ddq_max, ddq_c < ddq_min)
                print("accel outside limits")
            dddq_c = (ddq_c - ddq_c_old)/dt
            if np.any(dddq_c > 1.5*dddq_max) or np.any(dddq_c < 1.5*dddq_min) :
                #print(dddq_c, dddq_max, dddq_min, dddq_c[dddq_c > dddq_max], dddq_c < dddq_min) 
                print("jerk outside limits")
            q_c = np.clip(dq_c_old*dt + ddq_c*(dt**2)/2 + q_c_old, q_min,q_max)
                
        
        t_sim = t_sim+1
        
        # data.e_pos_list_ruckig.append(np.linalg.norm(V23@(X_c.t-X_init.translation)))
        data.e_pos_list_ruckig.append(np.linalg.norm((X_c.translation-X_init.translation-np.array(data.x_list[-1]*u[:3]))))
        data.e_rot_list_ruckig.append(np.linalg.norm(pin.log3(X_init.rotation.T@X_c.rotation)))
        
    print(f'Calculation duration: {time.time()-s} [s]')
    data.t_ruckig = np.array(range(0,len(data.x_list)))*dt
    print(f'Trajectory duration: {data.t_ruckig[-1]:0.4f} [s]')
    return data
    


def rand_num(delta):
    return np.random.rand()*2*delta - delta

def find_random_poses_with_distance(distance, robot, q0, iterations = 10, joint_limits=True, verify_line = False , n_waypoints = 10 ,angle =np.pi/6):
    X_r = robot.fkine(q0)
    found = False
    while not found:
        print("searching")
        X_init = None
        while X_init is None or np.linalg.norm(robot.fkine(robot.ikine_LM(X_init, q0, joint_limits=joint_limits).q).t-X_init.t) > 1e-5:
            X_init = SE3(np.random.rand(1,3)*distance-0.5*distance+np.array([0.5,0,0.4]))*SE3(SO3(X_r.R))*SE3(SO3.Rx(rand_num(angle)))*SE3(SO3.Ry(rand_num(angle)))

        X_final = None
        i = 0
        while (X_final is None or np.linalg.norm(robot.fkine(robot.ikine_LM(X_final, q0, joint_limits=joint_limits).q).t-X_final.t) > 1e-5) and i <= iterations:
            print("not attainable final")
            v= np.random.rand(3)*2-1
            v = v/np.linalg.norm(v)*distance
            X_final = SE3(X_init.t + v)*SE3(SO3(X_init.R))
            i = i+1
        if i < iterations:
            found = True

        if verify_line:
            q_line = [ robot.ikine_LM(X_init,q0,joint_limits=joint_limits).q ]
            X_i = np.linspace(X_init.t, X_final.t, n_waypoints)
            print(n_waypoints)
            for x in X_i[1:]:
                T = SE3(x)*SE3(SO3(X_init.R))
                sol = robot.ikine_LM(T,q_line[-1],joint_limits=joint_limits)         # solve IK
                X_found = robot.fkine(sol.q)
                if np.linalg.norm(robot.fkine(robot.ikine_LM(X_found, joint_limits=joint_limits).q).t-T.t) > 1e-5:
                    break
                q_line.append(sol.q)    
            
            if len(q_line) < len(X_i):
                print("staright line not possible")
                continue
    
    
    
    return X_init, X_final, q_line


def find_random_poses_with_distance_pinocchio(distance, robot, q0, iterations = 10, joint_limits=True, verify_line = False , n_waypoints = 10,angle =np.pi/6):
    X_r = robot.forward(q0)
    found = False
    while not found:
        print("searching")
        X_init = None
        while X_init is None or np.linalg.norm(robot.forward(robot.ik(X_init, q0, qlim=joint_limits, verbose=False)).translation-X_init.translation) > 1e-3:
            X_init = pin.SE3(X_r.rotation, (np.random.rand(1,3)*distance-0.5*distance+np.array([0.5,0,0.4])).reshape((3,)))
            x,y = rand_num(angle),rand_num(angle)
            X_init = X_init*pin.SE3(pin.rpy.rpyToMatrix(x, y,0), np.zeros(3))

        X_final = None
        i = 0
        while (X_final is None or np.linalg.norm(robot.forward(robot.ik(X_final, q0, qlim=joint_limits, verbose=False)).translation-X_final.translation) > 1e-3) and i <= iterations:
            print("not attainable final")
            v= np.random.rand(3)*2-1
            v = v/np.linalg.norm(v)*distance
            X_final = pin.SE3(X_init.rotation, X_init.translation + v)
            i = i+1
            
        if i < iterations:
            found = True
        
        if verify_line:
            print("verify straitgh line")
            q_line = [ robot.ik(X_init,q0,qlim=joint_limits, verbose=False) ]
            X_i = np.linspace(X_init.translation, X_final.translation, n_waypoints)
            print(n_waypoints)
            for x in X_i[1:]:
                T = pin.SE3(X_init.rotation, x)
                q = robot.ik(T, q_line[-1], qlim=joint_limits, verbose=False)         # solve IK
                X_found = robot.forward(q)
                if np.linalg.norm(robot.forward(robot.ik(X_found, qlim=joint_limits, verbose=False)).translation-T.translation) > 1e-3:
                    print(robot.forward(robot.ik(X_found, qlim=joint_limits, verbose=False)).translation, T.translation)
                    break
                q_line.append(q)    
            
            if len(q_line) < len(X_i):
                print("staright line not possible")
                found = False
                continue
    
    return X_init, X_final, q_line


def simulate_toppra(X_init, X_final, ts, qs, qds, qdds, q0,  lims, scale, robot = None, options=None , robot_pyn = None ):
    s =  time.time()
    data = TrajData()
    
    print('simulating trajectory')
    
    # check the type of the X_init and X_final
    if type(X_init) == spatialmath.pose3d.SE3:
        X_init = pin.SE3(X_init.R, X_init.t)
        X_final = pin.SE3(X_final.R, X_final.t)
    elif type(X_init) == np.ndarray:
        X_init = pin.SE3(X_init[:3,:3], X_init[:3,3])
        X_final = pin.SE3(X_final[:3,:3], X_final[:3,3])
    elif type(X_init) == pin.SE3:
        pass
    else:
        print("Unknown type of X_init")
        return
    
    u = np.zeros(6)
    u[:3] = X_final.translation-X_init.translation
    u = u/np.linalg.norm(u)
    
    # limtis calculation
    dq_max = scale*lims['dq_max']
    ddq_max = scale*lims['ddq_max']
    dddq_max = scale*lims['dddq_max']
    q_min, q_max = lims['q_min'], lims['q_max']
    dq_min = -dq_max
    ddq_min = -ddq_max
    dddq_min = -dddq_max
    
    if options is not None and 'calculate_limits' in options.keys():
        calculate_limits = options['calculate_limits']
    else: 
        calculate_limits = True
        
        
    if calculate_limits:
        U,S,V = np.linalg.svd(u[:,None])
        V2 = U[:,1:].T
        
        data.ds_max_list=[]
        data.dds_max_list=[]
        data.ddds_max_list=[]
        data.ds_min_list=[]
        data.dds_min_list=[]
        data.ddds_min_list=[]
    
    data.x3d_top = []
    data.x_top = []
    data.dx_top = []
    data.ddx_top = []
    data.dddx_top = []
    data.qr_list = []
    
    U,S,V = np.linalg.svd(u[:3,None])
    V23 = U[:,1:].T
    data.e_pos_list, data.e_rot_list = [], []
    
    q_c = qs[0]
    
    if robot_pyn is not None:
        dq_c, ddq_c, dddq_c =  np.zeros(robot_pyn.n), np.zeros(robot_pyn.n), np.zeros(robot_pyn.n)
    elif robot is not None:
        dq_c, ddq_c, dddq_c =  np.zeros(robot.n), np.zeros(robot.n), np.zeros(robot.n)
    else:
        print("No robot model provided")
        return
    
    t_last = 0
    for t, q,qd,qdd in zip(ts, qs,qds,qdds):
        
        
        if robot_pyn is not None:
            X_c = robot_pyn.forward(q_c)
            X_dc = robot_pyn.forward(q)
            J =robot_pyn.jacobian(q_c)
            J_dot = robot_pyn.jacobian_dot(q_c,dq_c)    
            x_t = X_c.translation
            data.x3d_top.append(x_t)
            data.x_top.append(u[:3]@(x_t-X_init.translation))
            c = u@J
            c_dot = u@J_dot
            data.dx_top.append(c@dq_c)
            data.ddx_top.append(c@ddq_c + c_dot@dq_c)
            data.dddx_top.append(c@dddq_c+ c_dot@ddq_c)
            
            data.qr_list.append(q_c)
            
            data.e_pos_list.append(np.linalg.norm((X_c.translation-X_dc.translation)))
            # data.e_pos_list.append(np.linalg.norm(V23@(X_c.t-X_init.t)))
            data.e_rot_list.append(np.linalg.norm(pin.log3(X_init.rotation.T@X_c.rotation)))
            
        elif robot is not None:
            X_c = robot.fkine(q_c)
            X_dc = robot.fkine(q)
            J =robot.jacob0(q_c)
            J_dot = robot.jacob0_dot(q_c,dq_c)
                
            x_t = X_c.t
            data.x3d_top.append(x_t)
            data.x_top.append(u[:3]@(x_t-X_init.translation))
            c = u@J
            c_dot = u@J_dot
            data.dx_top.append(c@dq_c)
            data.ddx_top.append(c@ddq_c + c_dot@dq_c)
            data.dddx_top.append(c@dddq_c+ c_dot@ddq_c)
            
            data.qr_list.append(q_c)
            
            data.e_pos_list.append(np.linalg.norm((X_c.t-X_dc.t)))
            # data.e_pos_list.append(np.linalg.norm(V23@(X_c.t-X_init.t)))
            data.e_rot_list.append(np.linalg.norm(pin.log3(X_init.rotation.T@X_c.R)))
        
        dt = t - t_last
        t_last = t
        if not dt: 
            dt =0.001
    
        # calculate position + limit
        q_c = q #dq_c*dt + q_c #+ ddq_c*(dt**2)/2 + q_c
        q_c = np.clip(q_c, q_min, q_max)
        
        dq_c_old = dq_c
        ddq_c_old = ddq_c
        # calculate velocity + limit
        # dq_c = qd #dq_c + ddq_c*dt
        dq_c = qd#np.clip(qd, dt*ddq_min+dq_c, dt*ddq_max+dq_c) 
        dq_c = np.clip(dq_c, dq_min, dq_max)
        # limlit jerk
        ddq_c = qdd#np.clip((dq_c-dq_c_old)/dt, dt*dddq_min+ddq_c, dt*dddq_max+ddq_c) 
        # limlit acceleration
        # ddq_c = qdd
        ddq_c = np.clip(ddq_c, ddq_min, ddq_max) 
        # calculate jerk
        dddq_c = (ddq_c - ddq_c_old)/dt
        
        
        
        if calculate_limits:
            Aeq = V2@J
            beq = np.zeros(5)
            try:
                data.ds_max_list.append(c@solve_lp(-c, A_eq=Aeq, b_eq=beq, x_min=dq_min, x_max=dq_max))
                data.dds_max_list.append(c@solve_lp(-c, A_eq=Aeq, b_eq=-V2@J_dot@dq_c, x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                data.ddds_max_list.append(c@solve_lp(-c, A_eq=Aeq, b_eq=-V2@J_dot@ddq_c, x_min=dddq_min, x_max=dddq_max) + u@J_dot@ddq_c)
                data.ds_min_list.append(c@solve_lp(c, A_eq=Aeq, b_eq=beq, x_min=dq_min, x_max=dq_max))
                data.dds_min_list.append(c@solve_lp(c, A_eq=Aeq, b_eq=-V2@J_dot@dq_c, x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                data.ddds_min_list.append(c@solve_lp(c, A_eq=Aeq, b_eq=-V2@J_dot@ddq_c, x_min=dddq_min, x_max=dddq_max) + u@J_dot@ddq_c)
            except:
                print("except dds")
                data.ds_max_list.append(c@solve_lp(-c, A_eq=Aeq, b_eq=np.zeros(5), x_min=dq_min, x_max=dq_max))
                data.dds_max_list.append(c@solve_lp(-c, A_eq=Aeq, b_eq=np.zeros(5), x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                data.ddds_max_list.append(c@solve_lp(-c, A_eq=Aeq, b_eq=np.zeros(5), x_min=dddq_min, x_max=dddq_max) + u@J_dot@ddq_c)
                data.ds_min_list.append(c@solve_lp(c, A_eq=Aeq, b_eq=np.zeros(5), x_min=dq_min, x_max=dq_max))
                data.dds_min_list.append(c@solve_lp(c, A_eq=Aeq, b_eq=np.zeros(5), x_min=ddq_min, x_max=ddq_max) + u@J_dot@dq_c)
                data.ddds_min_list.append(c@solve_lp(c, A_eq=Aeq, b_eq=np.zeros(5), x_min=dddq_min, x_max=dddq_max) + u@J_dot@ddq_c)
    
    data.t_toppra = ts
    print('TOPPRA trajecotry simulation time',time.time() - s)
    print(f'Trajectory duration: {data.t_toppra[-1]:0.4f} [s]')
    return data


import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo

def caclulate_toppra_trajectory(X_init, X_final,  q0, lims, scale, d_waypoint, robot=None, robot_pyn=None, data=None):
    s = time.time()
    # ta.setup_logging("INFO")
    
    
    if robot_pyn is not None:
        # check the type of the X_init and X_final
        if type(X_init) == spatialmath.pose3d.SE3:
            X_init = pin.SE3(X_init.R, X_init.t)
            X_final = pin.SE3(X_final.R, X_final.t)
        elif type(X_init) == np.ndarray:
            X_init = pin.SE3(X_init[:3,:3], X_init[:3,3])
            X_final = pin.SE3(X_final[:3,:3], X_final[:3,3])
        elif type(X_init) == pin.SE3:
            pass
        else:
            print("Unknown type of X_init")
            return
    
    # limtis calculation
    dq_max = scale*lims['dq_max']
    ddq_max = scale*lims['ddq_max']
    
    # calculate the waypoints
    if robot_pyn is not None:
        d = np.linalg.norm(X_final.translation-X_init.translation)
    else:
        d = np.linalg.norm(X_final.t-X_init.t)
    #number of waypoints in joint space
    n_waypoints  = int(d/d_waypoint)
    # print(f'traj length: {d}, number of waypoints: {n_waypoints}')
    
    if data is not None:
        print('using provided data')
        x_np = np.array(data.x_list)
        x_v = np.linspace(0,x_np[-1], n_waypoints)
        inds = []
        for x_wp in x_v:
            inds.append(np.where(x_np >= x_wp)[0][0])
        q_line = [data.qr_list[int(i)]  for i in inds]
        
    else:
        if robot is not None:
            # print('calculation waypoints')
            X_i = np.linspace(X_init.t,X_final.t,n_waypoints)
            q_line = [ robot.ikine_LM(X_init, q0).q ]
            for x in X_i[1:]:
                T = SE3(x)*SE3(SO3(X_init.R))
                sol = robot.ikine_LM(T,q_line[-1])#,joint_limits=true)         # solve IK
                q_line.append(sol.q)
        else:
            # print('calculation waypoints')
            X_i = np.linspace(X_init.translation,X_final.translation,n_waypoints)
            q_line = [ robot_pyn.ik(X_init, q0, qlim=True, verbose=False) ]
            for x in X_i[1:]:
                T= pin.SE3(X_init.rotation, x)
                q_line.append(robot_pyn.ik(T,q_line[-1], qlim=True, verbose=False))
            

        
    # print("Waypoints calculation time:",time.time()-s)
    s1 = time.time()
    # calculate the traectory
    ss = np.linspace(0,1,len(q_line))
    path = ta.SplineInterpolator(ss, q_line)
    pc_vel = constraint.JointVelocityConstraint(dq_max)
    pc_acc = constraint.JointAccelerationConstraint(ddq_max)
    instance = algo.TOPPRA([pc_vel, pc_acc], path, parametrizer="ParametrizeConstAccel")
    jnt_traj = instance.compute_trajectory()
    
    ts_sample = np.arange(0, jnt_traj.duration, 0.001)
    qs_sample = jnt_traj(ts_sample)
    qds_sample = jnt_traj(ts_sample, 1)
    qdds_sample = jnt_traj(ts_sample, 2)
    
    # print("TOPPRA calculation time:",time.time()-s1)
    # print("Waypoints+TOPPRA calculation time:",time.time()-s)
    return ts_sample, qs_sample, qds_sample, qdds_sample
    # return time.time()-s, time.time()-s1
