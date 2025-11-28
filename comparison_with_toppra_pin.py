#!/usr/bin/env python3
"""
Online approach to near time-optimal task-space trajectory planning

Comparison of Robot's movement capacity aware real-time trajectory planning 
in Cartesian Space with TOPPRA
"""

import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plots
import matplotlib.pyplot as plt

import numpy as np
import time

import pinocchio as pin
from example_robot_data import load
from planning_utils import *

import meshcat
from pinocchio.visualize import MeshcatVisualizer
from meshcat_shapes import *

from pynocchio import RobotWrapper


def main():
    # ========================================================================
    # USER INPUT
    # ========================================================================
    print("=" * 70)
    print("Robot Capacity Aware Trajectory Planning Comparison")
    print("=" * 70)
    
    # Get capacity scale from user
    while True:
        try:
            scale_input = input("\nEnter robot capacity scale (0.0-1.0, default=0.5): ").strip()
            if scale_input == "":
                scale = 0.5
            else:
                scale = float(scale_input)
            if 0.0 <= scale <= 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0")
    
    # Get trajectory length from user
    while True:
        try:
            traj_input = input("\nEnter trajectory length in meters (0.0-1.0, default=0.8): ").strip()
            if traj_input == "":
                traj_length = 0.8
            else:
                traj_length = float(traj_input)
            if 0.0 < traj_length <= 1.0:
                break
            else:
                print("Please enter a value between 0.0 and 1.0")
        except ValueError:
            print("Invalid input. Please enter a number between 0.0 and 1.0")
    
    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Capacity scale: {scale}")
    print(f"  Trajectory length: {traj_length} m")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # SETUP ROBOT
    # ========================================================================
    print("Loading robot model...")
    panda_pin = load('panda')
    panda_pin.data = panda_pin.model.createData()
    
    viewer = meshcat.Visualizer()
    viewer.open()
    
    panda_tip_pin = "panda_hand_tcp"
    
    panda_pyn = RobotWrapper(
        robot_wrapper=panda_pin, 
        tip=panda_tip_pin, 
        open_viewer=False, 
        start_visualisation=True, 
        viewer=viewer, 
        fix_joints=[
            panda_pin.model.getJointId("panda_finger_joint1"),
            panda_pin.model.getJointId("panda_finger_joint2")
        ]
    )
    viz = panda_pyn.viz
    viz.display_collisions = False
    
    # ========================================================================
    # DEFINE CARTESIAN PATH
    # ========================================================================
    print("Generating random trajectory...")
    n_waypoints = int(traj_length / 0.05)
    
    q0 = (panda_pyn.model.upperPositionLimit + panda_pyn.model.lowerPositionLimit) / 2
    
    X_init, X_final, q_line = find_random_poses_with_distance_pinocchio(
        robot=panda_pyn, 
        distance=traj_length, 
        q0=q0, 
        verify_line=True, 
        n_waypoints=n_waypoints, 
        angle=np.pi/2
    )
    
    print(f'Initial point: {X_init}')
    print(f'Final point: {X_final}')
    print(f'Waypoints num: {n_waypoints}')
    print(f'Trajectory length: {traj_length}')
    
    display(viz, panda_pyn, panda_tip_pin, "end_effector_target", 
            panda_pyn.ik(X_init, qlim=True, verbose=False))
    display_frame(viz, "end_effector_start", X_init.np)
    display_frame(viz, "end_effector_end", X_final.np)
    
    # Visualize trajectory waypoints
    print("Displaying trajectory waypoints \nPress Ctrl+C to continue...")
    try:
        while True:
            for i, q in enumerate(q_line):
                display(viz, panda_pyn, panda_tip_pin, f"waypoint_{i}", q)
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    
    # ========================================================================
    # DEFINE ROBOT LIMITS
    # ========================================================================
    print("Setting up robot limits...")
    q_min, q_max = panda_pyn.model.lowerPositionLimit, panda_pyn.model.upperPositionLimit
    dq_max = panda_pyn.model.velocityLimit
    dq_min = -dq_max
    ddq_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20])
    ddq_min = -ddq_max
    dddq_max = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000])
    dddq_min = -dddq_max
    t_max = np.array([87, 87, 87, 87, 20, 20, 20])
    t_min = -t_max
    
    # Cartesian space limits
    dddx_max = np.array([6500.0, 6500.0, 6500.0])
    dddx_min = -dddx_max
    ddx_max = np.array([13.0, 13, 13])
    ddx_min = -ddx_max
    dx_max = np.array([1.7, 1.7, 1.7])
    dx_min = -dx_max
    
    limits = {
        'q_min': q_min, 'q_max': q_max, 
        'dq_max': dq_max, 'ddq_max': ddq_max, 
        'dddq_max': dddq_max, 't_max': t_max, 
        'dx_max': dx_max, 'ddx_max': ddx_max, 
        'dddx_max': dddx_max
    }
    
    # ========================================================================
    # COMPUTE TRAJECTORIES
    # ========================================================================
    print(f"\nComputing trajectory with capacity scale: {scale}")
    
    # Our approach
    print("Computing capacity-aware trajectory (ours)...")
    options = {
        'Kp': 600, 
        'Kd': 150, 
        'Ki': 0.0,
        'Tf': 0.01,             
        'uptate_current_position': True,
        'clamp_velocity': True, 
        'clamp_min_accel': True,
        'scaled_qp_limits': True,
        'override_acceleration': True, 
        'scale_limits': True,
        'calculate_limits': True,
        'downsampling_ratio': 1,
        'use_manip_grad': False,
        'manip_grad_w': 5000.0,
        'dt': 0.001,
        'qp_form': 'acceleration'
    }
    
    data = compute_capacity_aware_trajectory(
        X_init, X_final, 
        robot=None, 
        robot_pyn=panda_pyn,  
        lims=limits, 
        scale=scale, 
        options=options, 
        q0=(q_min + q_max) / 2
    )
    
    # TOPPRA
    print("Computing TOPPRA trajectory...")
    ts_sample, qs_sample, qds_sample, qdds_sample = caclulate_toppra_trajectory(
        X_init, X_final, 
        robot_pyn=panda_pyn, 
        q0=q0, 
        d_waypoint=0.05, 
        lims=limits, 
        scale=scale
    )
    
    data_top = simulate_toppra(
        X_init, X_final, 
        ts_sample, qs_sample, qds_sample, qdds_sample, 
        q0=q0, 
        robot_pyn=panda_pyn, 
        lims=limits, 
        scale=scale, 
        options=options
    )
    
    print(f"\nTOPPRA trajectory duration: {ts_sample[-1]:.3f} s")
    print(f"Ours trajectory duration: {data.t_ruckig[-1]:.3f} s")
    
    # ========================================================================
    # VISUALIZATION - DUAL ROBOT ANIMATION SETUP
    # ========================================================================
    print("\nSetting up dual robot visualization...")
    panda_ours = panda_pyn.robot
    panda_toppra = panda_pyn.robot
    
    viewer_dual = viewer
    
    panda_toppra.data = panda_toppra.model.createData()
    
    viz_l = panda_pyn.viz
    viz_r = MeshcatVisualizer(panda_toppra.model, panda_toppra.collision_model, panda_toppra.visual_model)
        
    viz_r.initViewer(open=False, viewer=viewer_dual)
    viz_r.loadViewerModel("toppra", color=[0.0, 0.0, 0.0, 0.5])
    
    # Shift right robot
    T_shift = np.eye(4)
    T_shift[1, 3] = 0.5  # 0.5 meter along y
    viewer_dual["toppra"].set_transform(T_shift)
    
    display(viz_l, panda_ours, panda_tip_pin, "end_effector_ruc", data.qr_list[0])
    display(viz_r, panda_toppra, panda_tip_pin, "end_effector_toppra", data_top.qr_list[0])
    
    display_frame(viz_l, "start", X_init.np)
    display_frame(viz_l, "end", X_final.np)
    display_frame(viz_r, "start1", T_shift @ X_init)
    display_frame(viz_r, "end1", T_shift @ X_final)
    
    # ========================================================================
    # PLOTS
    # ========================================================================
    print("\nGenerating plots...")
    
    # Plot 1: Time evolution of trajectory variables
    plt.rcParams['axes.facecolor'] = 'white'
    fig1, axs = plt.subplots(4, 2, sharex=True, figsize=[10, 8])
    
    linewidth = 2
    fontsize = 12
    plt.rcParams.update({'font.size': fontsize, 'xtick.color': 'black', 'ytick.color': 'black'})
    
    # Position
    axs[0, 0].plot(data.t_ruckig[1:], data.x_q_list[1:], '#6C8EBF', linewidth=linewidth)
    axs[0, 1].plot(data_top.t_toppra, data_top.x_top, '#B85450', linewidth=linewidth)
    
    # Velocity
    axs[1, 0].plot(data.t_ruckig[1:], data.dx_max_list[1:], 'k--', linewidth=linewidth)
    axs[1, 0].plot(data.t_ruckig[1:], data.dx_min_list[1:], 'k--', linewidth=linewidth)
    axs[1, 0].plot(data.t_ruckig[1:], data.dx_q_list[1:], '#6C8EBF', linewidth=linewidth)
    axs[1, 1].plot(data_top.t_toppra, data_top.dx_top, '#6C8EBF', linewidth=linewidth)
    axs[1, 1].plot(data_top.t_toppra, data_top.ds_max_list, 'k--', linewidth=linewidth)
    axs[1, 1].plot(data_top.t_toppra, data_top.ds_min_list, 'k--', linewidth=linewidth)
    
    # Acceleration
    axs[2, 0].plot(data.t_ruckig[1:], data.ddx_max_list[1:], 'k--', linewidth=linewidth)
    axs[2, 0].plot(data.t_ruckig[1:], data.ddx_min_list[1:], 'k--', linewidth=linewidth)
    axs[2, 0].plot(data.t_ruckig[1:], data.ddx_q_list[1:], '#6C8EBF', linewidth=linewidth)
    axs[2, 1].plot(data_top.t_toppra, data_top.ddx_top, '#B85450', linewidth=linewidth)
    axs[2, 1].plot(data_top.t_toppra, data_top.dds_max_list, 'k--', linewidth=linewidth)
    axs[2, 1].plot(data_top.t_toppra, data_top.dds_min_list, 'k--', linewidth=linewidth)
    
    # Jerk
    axs[3, 0].plot(data.t_ruckig[1:], data.dddx_max_list[1:], 'k--', linewidth=linewidth)
    axs[3, 0].plot(data.t_ruckig[1:], data.dddx_min_list[1:], 'k--', linewidth=linewidth)
    axs[3, 0].plot(data.t_ruckig[1:], data.dddx_q_list[1:], '#6C8EBF', linewidth=linewidth)
    axs[3, 1].plot(data_top.t_toppra, data_top.dddx_top, '#B85450', linewidth=linewidth)
    axs[3, 1].plot(data_top.t_toppra, data_top.ddds_max_list, 'k--', linewidth=linewidth)
    axs[3, 1].plot(data_top.t_toppra, data_top.ddds_min_list, 'k--', linewidth=linewidth)
    
    axs[0, 0].set_title("ours")
    axs[0, 0].set_ylabel(r'${s}$ $[m]$', color="black")
    axs[1, 0].set_ylabel(r'$\dot{s}$ $[m/s]$', color="black")
    axs[2, 0].set_ylabel(r'$\ddot{s}$ $[m/s^2]$', color="black")
    axs[3, 0].set_ylabel(r'$\dddot{s}$ $[m/s^3]$', color="black")
    axs[3, 0].set_xlabel(r'time $[s]$', color="black")
    axs[3, 1].set_xlabel(r'time $[s]$', color="black")
    axs[0, 1].set_title("topp-ra")
    
    plt.tight_layout()
    
    # Plot 2: Comparison
    fig2, axs2 = plt.subplots(1, 4, sharex=True, figsize=[12, 3])
    plt.rcParams['axes.facecolor'] = 'white'
    linw = 2
    plt.rcParams.update({'font.size': 12, 'xtick.color': 'black', 'ytick.color': 'black'})
    
    axs2[0].plot(data.t_ruckig[1:], data.x_list[1:], linewidth=linw, color='#6C8EBF', label='ours')
    axs2[0].plot(ts_sample, data_top.x_top, linewidth=linw, color='#B85450', label='topp-ra')
    axs2[0].set_xlabel(r'time $[s]$', color="black")
    axs2[0].set_ylabel(r'${s}$ $[m]$', color="black")
    axs2[0].legend(loc='best')
    
    axs2[1].plot(data_top.t_toppra, data_top.dx_top, linewidth=linw, color='#B85450', label='topp-ra')
    axs2[1].plot(data_top.t_toppra[1:], data_top.ds_max_list[1:], color='#B85450', linestyle="--", linewidth=linw, label='topp-ra limits')
    axs2[1].plot(data.t_ruckig[1:], data.dx_q_list[1:], linewidth=linw, color='#6C8EBF', label='ours')
    axs2[1].plot(data.t_ruckig[1:], data.dx_max_list[1:], color='#6C8EBF', linestyle="--", linewidth=linw, label='ours limits')
    axs2[1].set_xlabel(r'time $[s]$', color="black")
    axs2[1].set_ylabel(r'$\dot{s}$ $[m/s]$', color="black")
    axs2[1].legend(loc='best')
    
    axs2[2].plot(data_top.t_toppra, data_top.ddx_top, linewidth=linw, color='#B85450', label='topp-ra')
    axs2[2].plot(data_top.t_toppra[1:], data_top.dds_max_list[1:], color='#B85450', linestyle="--", linewidth=linw, label='topp-ra limits')
    axs2[2].plot(data_top.t_toppra[1:], data_top.dds_min_list[1:], color='#B85450', linestyle="--", linewidth=linw)
    axs2[2].plot(data.t_ruckig[1:], data.ddx_q_list[1:], linewidth=linw, color='#6C8EBF', label='ours')
    axs2[2].plot(data.t_ruckig[1:], data.ddx_max_list[1:], color='#6C8EBF', linestyle="--", linewidth=linw, label='ours limits')
    axs2[2].plot(data.t_ruckig[1:], data.ddx_min_list[1:], color='#6C8EBF', linestyle="--", linewidth=linw)
    axs2[2].set_xlabel(r'time $[s]$', color="black")
    axs2[2].set_ylabel(r'$\ddot{s}$ $[m/s^2]$', color="black")
    
    axs2[3].plot(data_top.t_toppra, np.array(data_top.dddx_top) / 1000, linewidth=linw, color='#B85450', label='topp-ra')
    axs2[3].plot(data_top.t_toppra[1:], np.array(data_top.ddds_max_list[1:]) / 1000, color='#B85450', linestyle="--", linewidth=linw, label='ours limits')
    axs2[3].plot(data_top.t_toppra[1:], np.array(data_top.ddds_min_list[1:]) / 1000, color='#B85450', linestyle="--", linewidth=linw)
    axs2[3].plot(data.t_ruckig[1:], np.array(data.dddx_q_list[1:]) / 1000, linewidth=linw, color='#6C8EBF', label='ours')
    axs2[3].plot(data.t_ruckig[1:], np.array(data.dddx_max_list[1:]) / 1000, color='#6C8EBF', linestyle="--", linewidth=linw, label='ours limits')
    axs2[3].plot(data.t_ruckig[1:], np.array(data.dddx_min_list[1:]) / 1000, color='#6C8EBF', linestyle="--", linewidth=linw)
    axs2[3].set_xlabel(r'time $[s]$', color="black")
    axs2[3].set_ylabel(r'$\dddot{s}$   1000x$[m/s^3]$', color="black")
    
    plt.tight_layout()
    
    # Plot 3: Tracking error
    fig3, ax3 = plt.subplots(1, 2, sharex=True, figsize=[12, 5])
    ax3[0].plot(data.t_ruckig, np.array(data.e_pos_list_ruckig) * 1e3, '#6C8EBF', linewidth=linewidth, label="ours")
    ax3[0].plot(data_top.t_toppra, np.array(data_top.e_pos_list) * 1e3, '#B85450', linewidth=linewidth, label="topp-ra")
    ax3[1].plot(data.t_ruckig, np.rad2deg(data.e_rot_list_ruckig), '--', color="#6C8EBF", linewidth=linewidth, label="ours")
    ax3[1].plot(data_top.t_toppra, np.rad2deg(data_top.e_rot_list), 'r--', color="#B85450", linewidth=linewidth, label="topp-ra")
    
    ax3[0].set_title("position error[mm]")
    ax3[0].set_ylabel(r'error $[mm]$', color="black")
    ax3[1].set_ylabel(r'error $[deg]$', color="black")
    ax3[1].set_title("orientation error [deg]")
    ax3[0].set_xlabel(r'time $[s]$', color="black")
    ax3[1].set_xlabel(r'time $[s]$', color="black")
    ax3[0].legend()
    ax3[1].legend()
    
    plt.tight_layout()
    
    # Show all plots without blocking
    print("\nDisplaying plots...")
    plt.ion()  # Turn on interactive mode
    plt.show()
    plt.pause(2.0)  # Allow plots to render
    
    # ========================================================================
    # ANIMATE ROBOTS
    # ========================================================================
    print("\nAnimating robots (LEFT: ours, RIGHT: toppra)...")
    print("Press Ctrl+C to stop animation\n")
    
    t0_plot = time.time()
    
    try:
        # Animation loop - runs continuously
        while True:
            t0 = time.time()
            t_max = max(data.t_ruckig[-1], data_top.t_toppra[-1])
            
            # Precompute indices for faster lookup
            last_ind_r = 0
            last_ind_t = 0
            
            while (time.time() - t0) < t_max:
                t_current = time.time() - t0
                
                # Find indices more efficiently - start from last position
                # TOPPRA index
                ind_t = last_ind_t
                while ind_t < len(data_top.t_toppra) - 1 and data_top.t_toppra[ind_t] <= t_current:
                    ind_t += 1
                last_ind_t = ind_t
                
                # Ours index
                ind_r = last_ind_r
                while ind_r < len(data.t_ruckig) - 1 and data.t_ruckig[ind_r] <= t_current:
                    ind_r += 1
                last_ind_r = ind_r
                
                # Update robot poses
                display(viz_l, panda_ours, panda_tip_pin, "end_effector_ruc", data.qr_list[ind_r])
                display(viz_r, panda_toppra, panda_tip_pin, "end_effector_toppra", qs_sample[ind_t], T_shift)
                
                # Update plots less frequently (every 10 frames)
                if time.time() - t0_plot > 5.0:
                    t0_plot = time.time()
                    plt.pause(0.5)
    
    except KeyboardInterrupt:
        print("\nAnimation stopped by user.")
    
    print("\nAnimation complete. Plots will remain open.")
    print("Close the plot windows to exit.")
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Block to keep plots open

    
    
    print("\n" + "=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
