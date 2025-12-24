#!/usr/bin/env python3
"""
EKF Position-Only Model Results Plotter

Generates comprehensive plots for EKF evaluation:
1. Position comparison (Ground Truth vs Noisy vs EKF)
2. Estimation error over time
3. RMSE bar chart comparison
4. Covariance evolution

"""

import sys
import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def find_latest_log(base_dir="/home/abithan_ubuntu/ros2_ws4/src/log_files"):
    """Find the most recent velocity log directory."""
    log_dirs = glob.glob(os.path.join(base_dir, "ekf_velocity_*"))
    if not log_dirs:
        raise FileNotFoundError(f"No velocity log directories found in {base_dir}")
    
    # Sort by modification time
    latest = max(log_dirs, key=os.path.getmtime)
    return latest


def load_data(log_input):
    """Load CSV data from log directory or file."""
    if os.path.isfile(log_input):
        csv_path = log_input
    else:
        csv_path = os.path.join(log_input, 'ekf_velocity_data.csv')
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} data points from {csv_path}")
    print(f"Columns: {list(df.columns)}")
    return df, os.path.dirname(csv_path)


def calculate_rmse(df):
    """Calculate RMSE for position."""
    rmse_noisy = []
    rmse_ekf = []
    
    for i in range(1, 4):
        # Noisy RMSE
        noisy_error = df[f'noisy_pos{i}'] - df[f'gt_pos{i}']
        rmse_noisy.append(np.sqrt(np.mean(noisy_error**2)))
        
        # EKF RMSE
        ekf_error = df[f'ekf_pos{i}'] - df[f'gt_pos{i}']
        rmse_ekf.append(np.sqrt(np.mean(ekf_error**2)))
    
    return rmse_noisy, rmse_ekf


def plot_results(df, output_dir):
    """Generate all plots in separate files."""
    time = df['timestamp'].values
    colors = ['#E74C3C', '#3498DB', '#2ECC71']
    joint_labels = ['Joint 1 (Base)', 'Joint 2 (Shoulder)', 'Joint 3 (Elbow)']

    for idx in range(3):
        i = idx + 1
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])
        
        # --- Position Tracking ---
        ax1 = axes[0]
        ax1.plot(time, df[f'gt_pos{i}'], 'k-', linewidth=2, label='Ground Truth')
        ax1.plot(time, df[f'noisy_pos{i}'], 'r.', alpha=0.4, markersize=3, label='Noisy Measurement (z_t)')
        ax1.plot(time, df[f'ekf_pos{i}'], 'b-', linewidth=1.5, label='EKF Estimate (μ_t)')
        
        ax1.set_xlabel('Time (s)', fontsize=11)
        ax1.set_ylabel('Position (rad)', fontsize=11)
        ax1.set_title(f'{joint_labels[idx]} - Position Tracking', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper left', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Calculate RMSE for this joint
        noisy_error = df[f'noisy_pos{i}'] - df[f'gt_pos{i}']
        ekf_error = df[f'ekf_pos{i}'] - df[f'gt_pos{i}']
        rmse_noisy = np.sqrt(np.mean(noisy_error**2))
        rmse_ekf = np.sqrt(np.mean(ekf_error**2))
        improvement = (rmse_noisy - rmse_ekf) / rmse_noisy * 100
        
        # Add RMSE text box
        textstr = f'RMSE Noisy: {rmse_noisy:.4f} rad\nRMSE EKF: {rmse_ekf:.4f} rad\nImprovement: {improvement:+.1f}%'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        # --- Covariance Evolution ---
        ax2 = axes[1]
        if f'cov_pos{i}' in df.columns:
            ax2.plot(time, df[f'cov_pos{i}'], color=colors[idx], linewidth=1.5)
            ax2.fill_between(time, df[f'cov_pos{i}'], alpha=0.3, color=colors[idx])
        
        ax2.set_xlabel('Time (s)', fontsize=11)
        ax2.set_ylabel('Covariance (Σ_t)', fontsize=11)
        ax2.set_title(f'{joint_labels[idx]} - Covariance Evolution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.suptitle(f'EKF Results - {joint_labels[idx]}\n'
                     f'Model: μ̄_t = μ_{{t-1}} + u_t · Δt', 
                     fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Save individual joint figure
        output_path = os.path.join(output_dir, f'ekf_joint{i}_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
        plt.close()
    
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    # --- Plot 1: All Joints Position Comparison ---
    ax1 = fig.add_subplot(gs[0, :])
    
    for i in range(1, 4):
        # Ground truth (solid line)
        ax1.plot(time, df[f'gt_pos{i}'], 
                 color=colors[i-1], linewidth=2, 
                 label=f'{joint_labels[i-1]} - Ground Truth')
        
        # Noisy measurements (dots)
        ax1.scatter(time[::3], df[f'noisy_pos{i}'].values[::3], 
                    color=colors[i-1], alpha=0.3, s=8, marker='.')
        
        # EKF estimates (dashed line)
        ax1.plot(time, df[f'ekf_pos{i}'], 
                 color=colors[i-1], linewidth=1.5, linestyle='--', alpha=0.8)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Position (rad)', fontsize=12)
    ax1.set_title('All Joints: Ground Truth (solid) vs Noisy z_t (dots) vs EKF μ_t (dashed)', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Position Error Over Time ---
    ax2 = fig.add_subplot(gs[1, 0])
    
    for i in range(1, 4):
        error = df[f'ekf_pos{i}'] - df[f'gt_pos{i}']
        ax2.plot(time, error, color=colors[i-1], linewidth=1.5, 
                 label=f'Joint {i}', alpha=0.8)
    
    ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_ylabel('Error (rad)', fontsize=11)
    ax2.set_title('EKF Position Estimation Error (μ_t - Ground Truth)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # --- Plot 3: Position RMSE Bar Chart ---
    ax3 = fig.add_subplot(gs[1, 1])
    
    rmse_noisy, rmse_ekf = calculate_rmse(df)
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, rmse_noisy, width, label='Noisy (z_t)', color='#E74C3C', alpha=0.8)
    bars2 = ax3.bar(x + width/2, rmse_ekf, width, label='EKF (μ_t)', color='#3498DB', alpha=0.8)
    
    # Add improvement annotations
    for j in range(3):
        improvement = (rmse_noisy[j] - rmse_ekf[j]) / rmse_noisy[j] * 100
        color = 'green' if improvement > 0 else 'red'
        arrow = '↓' if improvement > 0 else '↑'
        ax3.annotate(f'{abs(improvement):.1f}%{arrow}', 
                     xy=(x[j], max(rmse_noisy[j], rmse_ekf[j])),
                     xytext=(0, 5), textcoords='offset points',
                     ha='center', fontsize=10, fontweight='bold', color=color)
    
    ax3.set_xlabel('Joint', fontsize=11)
    ax3.set_ylabel('RMSE (rad)', fontsize=11)
    ax3.set_title('Position RMSE Comparison', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Joint 1', 'Joint 2', 'Joint 3'])
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)
    
    # Main title
    plt.suptitle('EKF Position Estimation Summary\n'
                 'Model: μ̄_t = g(u_t, μ_{t-1}) = μ_{t-1} + u_t · Δt (Constant Velocity, 360° in 10s)', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save summary figure
    output_path = os.path.join(output_dir, 'ekf_summary_results.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    
    plt.show()


def print_statistics(df):
    """Print detailed statistics."""
    print("\n" + "=" * 60)
    print("EKF POSITION ESTIMATION RESULTS")
    print("Model: Constant Velocity (Position Only)")
    print("State: [θ1, θ2, θ3]")
    print("=" * 60)
    
    rmse_noisy, rmse_ekf = calculate_rmse(df)
    
    print("\n--- Position RMSE Analysis ---")
    for i in range(3):
        improvement = (rmse_noisy[i] - rmse_ekf[i]) / rmse_noisy[i] * 100
        status = "✓ IMPROVED" if improvement > 0 else "✗ WORSE"
        print(f"Joint {i+1}: Noisy RMSE = {rmse_noisy[i]:.4f} rad, "
              f"EKF RMSE = {rmse_ekf[i]:.4f} rad, "
              f"Improvement = {improvement:+.1f}% {status}")
    
    # Overall RMSE
    overall_noisy = np.mean(rmse_noisy)
    overall_ekf = np.mean(rmse_ekf)
    overall_improvement = (overall_noisy - overall_ekf) / overall_noisy * 100
    
    print(f"\nOverall: Noisy RMSE = {overall_noisy:.4f} rad, "
          f"EKF RMSE = {overall_ekf:.4f} rad, "
          f"Improvement = {overall_improvement:+.1f}%")
    
    # Additional statistics
    print("\n--- Additional Statistics ---")
    duration = df['timestamp'].max() - df['timestamp'].min()
    print(f"Recording duration: {duration:.2f} seconds")
    print(f"Data points: {len(df)}")
    print(f"Sampling rate: {len(df)/duration:.1f} Hz")
    
    print("=" * 60)


def main():
    """Main function."""
    # Determine log path
    if len(sys.argv) > 1:
        log_input = sys.argv[1]
    else:
        try:
            log_input = find_latest_log()
            print(f"Using latest log: {log_input}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return 1
    
    # Load data
    try:
        df, output_dir = load_data(log_input)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Print statistics
    print_statistics(df)
    
    # Generate plots
    plot_results(df, output_dir)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
