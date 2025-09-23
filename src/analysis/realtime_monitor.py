#!/usr/bin/env python3
"""
Real-time Monitoring Dashboard for CFD Framework
================================================

This script provides real-time monitoring of CFD simulations including:
- Live plotting during simulation runs
- Residual convergence tracking
- Key metrics monitoring (drag, lift, flow rates)
- Auto-refresh capability for long runs

Requirements:
    numpy, matplotlib, pandas, watchdog

Usage:
    python realtime_monitor.py [options]
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import pandas as pd
import argparse
import glob
import os
import sys
import time
import threading
from pathlib import Path
import json
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class CFDMonitor:
    def __init__(self, watch_dir, output_dir="visualization_output"):
        self.watch_dir = watch_dir
        self.output_dir = output_dir
        self.monitoring = False
        self.data_history = []
        self.time_history = []
        self.metrics_history = {
            'max_velocity': [],
            'mean_velocity': [],
            'max_pressure': [],
            'mean_pressure': [],
            'total_kinetic_energy': [],
            'max_vorticity': [],
            'time': []
        }

        # Setup plotting
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('CFD Real-time Monitoring Dashboard', fontsize=16)
        self.axes = self.axes.flatten()

        # Initialize plots
        self.setup_plots()

        # File monitoring
        self.last_processed_file = None
        self.file_handler = VTKFileHandler(self)
        self.observer = Observer()

        os.makedirs(output_dir, exist_ok=True)

    def setup_plots(self):
        """Initialize all monitoring plots"""
        # 1. Velocity magnitude field (latest)
        self.ax_velocity = self.axes[0]
        self.ax_velocity.set_title('Latest Velocity Field')
        self.ax_velocity.set_xlabel('x (m)')
        self.ax_velocity.set_ylabel('y (m)')

        # 2. Pressure field (latest)
        self.ax_pressure = self.axes[1]
        self.ax_pressure.set_title('Latest Pressure Field')
        self.ax_pressure.set_xlabel('x (m)')
        self.ax_pressure.set_ylabel('y (m)')

        # 3. Max velocity vs time
        self.ax_max_vel = self.axes[2]
        self.ax_max_vel.set_title('Maximum Velocity vs Time')
        self.ax_max_vel.set_xlabel('Time Step')
        self.ax_max_vel.set_ylabel('Max Velocity (m/s)')
        self.ax_max_vel.grid(True, alpha=0.3)

        # 4. Mean velocity vs time
        self.ax_mean_vel = self.axes[3]
        self.ax_mean_vel.set_title('Mean Velocity vs Time')
        self.ax_mean_vel.set_xlabel('Time Step')
        self.ax_mean_vel.set_ylabel('Mean Velocity (m/s)')
        self.ax_mean_vel.grid(True, alpha=0.3)

        # 5. Kinetic energy vs time
        self.ax_energy = self.axes[4]
        self.ax_energy.set_title('Total Kinetic Energy vs Time')
        self.ax_energy.set_xlabel('Time Step')
        self.ax_energy.set_ylabel('Kinetic Energy')
        self.ax_energy.grid(True, alpha=0.3)

        # 6. Statistics summary
        self.ax_stats = self.axes[5]
        self.ax_stats.set_title('Current Statistics')
        self.ax_stats.axis('off')

        plt.tight_layout()

    def read_vtk_file(self, filename):
        """Read a VTK structured points file and extract data"""
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
        except (FileNotFoundError, PermissionError):
            return None

        # Parse header
        dimensions = None
        origin = None
        spacing = None
        data_start = None

        for i, line in enumerate(lines):
            if line.startswith('DIMENSIONS'):
                dimensions = [int(x) for x in line.split()[1:4]]
            elif line.startswith('ORIGIN'):
                origin = [float(x) for x in line.split()[1:4]]
            elif line.startswith('SPACING'):
                spacing = [float(x) for x in line.split()[1:4]]
            elif line.startswith('POINT_DATA'):
                data_start = i + 1
                break

        if not all([dimensions, origin, spacing, data_start]):
            return None

        nx, ny = dimensions[0], dimensions[1]
        x = np.linspace(origin[0], origin[0] + (nx-1)*spacing[0], nx)
        y = np.linspace(origin[1], origin[1] + (ny-1)*spacing[1], ny)

        # Parse data fields
        data_fields = {}
        field_names = ['u_velocity', 'v_velocity', 'pressure', 'density', 'temperature']

        i = data_start
        while i < len(lines):
            line = lines[i].strip()
            for field_name in field_names:
                if line.startswith(f'SCALARS {field_name}'):
                    i += 2  # Skip LOOKUP_TABLE line
                    values = []
                    while i < len(lines) and len(values) < nx * ny:
                        try:
                            row_values = [float(x) for x in lines[i].split()]
                            values.extend(row_values)
                        except (ValueError, IndexError):
                            break
                        i += 1
                    if len(values) == nx * ny:
                        data_fields[field_name] = np.array(values).reshape((ny, nx))
                    break
            else:
                i += 1

        if 'u_velocity' not in data_fields or 'v_velocity' not in data_fields:
            return None

        return {
            'x': x, 'y': y, 'data_fields': data_fields,
            'nx': nx, 'ny': ny, 'dx': spacing[0], 'dy': spacing[1],
            'filename': filename, 'timestamp': time.time()
        }

    def calculate_metrics(self, data):
        """Calculate flow metrics for monitoring"""
        fields = data['data_fields']
        u = fields['u_velocity']
        v = fields['v_velocity']
        pressure = fields.get('pressure', np.zeros_like(u))

        velocity_mag = np.sqrt(u**2 + v**2)

        # Calculate vorticity
        du_dy = np.gradient(u, data['dy'], axis=0)
        dv_dx = np.gradient(v, data['dx'], axis=1)
        vorticity = dv_dx - du_dy

        # Kinetic energy
        kinetic_energy = 0.5 * (u**2 + v**2)
        total_ke = np.sum(kinetic_energy) * data['dx'] * data['dy']

        metrics = {
            'max_velocity': np.max(velocity_mag),
            'mean_velocity': np.mean(velocity_mag),
            'max_pressure': np.max(pressure),
            'mean_pressure': np.mean(pressure),
            'total_kinetic_energy': total_ke,
            'max_vorticity': np.max(np.abs(vorticity)),
            'time': data['timestamp']
        }

        return metrics

    def update_plots(self, data):
        """Update all monitoring plots with new data"""
        if data is None:
            return

        # Calculate metrics
        metrics = self.calculate_metrics(data)

        # Update metrics history
        for key, value in metrics.items():
            if key in self.metrics_history:
                self.metrics_history[key].append(value)

        # Keep only last 100 points for performance
        max_history = 100
        for key in self.metrics_history:
            if len(self.metrics_history[key]) > max_history:
                self.metrics_history[key] = self.metrics_history[key][-max_history:]

        # Get current data
        fields = data['data_fields']
        u = fields['u_velocity']
        v = fields['v_velocity']
        pressure = fields.get('pressure', np.zeros_like(u))
        X, Y = np.meshgrid(data['x'], data['y'])
        velocity_mag = np.sqrt(u**2 + v**2)

        # Update velocity field
        self.ax_velocity.clear()
        cs1 = self.ax_velocity.contourf(X, Y, velocity_mag, levels=20, cmap='viridis')
        self.ax_velocity.set_title(f'Velocity Field - {os.path.basename(data["filename"])}')
        self.ax_velocity.set_xlabel('x (m)')
        self.ax_velocity.set_ylabel('y (m)')
        self.ax_velocity.set_aspect('equal')

        # Update pressure field
        self.ax_pressure.clear()
        cs2 = self.ax_pressure.contourf(X, Y, pressure, levels=20, cmap='plasma')
        self.ax_pressure.set_title(f'Pressure Field - {os.path.basename(data["filename"])}')
        self.ax_pressure.set_xlabel('x (m)')
        self.ax_pressure.set_ylabel('y (m)')
        self.ax_pressure.set_aspect('equal')

        # Update time series plots
        if len(self.metrics_history['max_velocity']) > 1:
            time_steps = range(len(self.metrics_history['max_velocity']))

            # Max velocity
            self.ax_max_vel.clear()
            self.ax_max_vel.plot(time_steps, self.metrics_history['max_velocity'], 'b-o', markersize=4)
            self.ax_max_vel.set_title('Maximum Velocity vs Time')
            self.ax_max_vel.set_xlabel('Time Step')
            self.ax_max_vel.set_ylabel('Max Velocity (m/s)')
            self.ax_max_vel.grid(True, alpha=0.3)

            # Mean velocity
            self.ax_mean_vel.clear()
            self.ax_mean_vel.plot(time_steps, self.metrics_history['mean_velocity'], 'r-o', markersize=4)
            self.ax_mean_vel.set_title('Mean Velocity vs Time')
            self.ax_mean_vel.set_xlabel('Time Step')
            self.ax_mean_vel.set_ylabel('Mean Velocity (m/s)')
            self.ax_mean_vel.grid(True, alpha=0.3)

            # Kinetic energy
            self.ax_energy.clear()
            self.ax_energy.plot(time_steps, self.metrics_history['total_kinetic_energy'], 'g-o', markersize=4)
            self.ax_energy.set_title('Total Kinetic Energy vs Time')
            self.ax_energy.set_xlabel('Time Step')
            self.ax_energy.set_ylabel('Kinetic Energy')
            self.ax_energy.grid(True, alpha=0.3)

        # Update statistics
        self.ax_stats.clear()
        self.ax_stats.axis('off')

        stats_text = f"""
        Current Flow Statistics:

        Velocity:
        Max: {metrics['max_velocity']:.4f} m/s
        Mean: {metrics['mean_velocity']:.4f} m/s

        Pressure:
        Max: {metrics['max_pressure']:.4f}
        Mean: {metrics['mean_pressure']:.4f}

        Energy:
        Total KE: {metrics['total_kinetic_energy']:.4f}

        Vorticity:
        Max |ω|: {metrics['max_vorticity']:.4f} 1/s

        Grid: {data['nx']} × {data['ny']}
        File: {os.path.basename(data['filename'])}
        Time: {time.strftime('%H:%M:%S', time.localtime(metrics['time']))}

        Monitoring Status: {'Active' if self.monitoring else 'Stopped'}
        Files Processed: {len(self.metrics_history['max_velocity'])}
        """

        self.ax_stats.text(0.05, 0.95, stats_text, transform=self.ax_stats.transAxes,
                          fontsize=10, verticalalignment='top', fontfamily='monospace',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))

        # Calculate convergence trends if we have enough data
        if len(self.metrics_history['max_velocity']) > 5:
            recent_max_vel = self.metrics_history['max_velocity'][-5:]
            recent_mean_vel = self.metrics_history['mean_velocity'][-5:]

            max_vel_trend = np.polyfit(range(5), recent_max_vel, 1)[0]
            mean_vel_trend = np.polyfit(range(5), recent_mean_vel, 1)[0]

            convergence_text = f"""
            Convergence Analysis:
            Max Velocity Trend: {max_vel_trend:+.6f}/step
            Mean Velocity Trend: {mean_vel_trend:+.6f}/step

            Status: {'Converging' if abs(max_vel_trend) < 1e-5 else 'Still Changing'}
            """

            self.ax_stats.text(0.05, 0.45, convergence_text, transform=self.ax_stats.transAxes,
                              fontsize=9, verticalalignment='top', fontfamily='monospace',
                              bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.8))

        plt.tight_layout()
        plt.draw()

    def process_new_file(self, filepath):
        """Process a new VTK file"""
        print(f"Processing new file: {os.path.basename(filepath)}")

        data = self.read_vtk_file(filepath)
        if data:
            self.update_plots(data)
            self.last_processed_file = filepath

            # Save metrics to CSV
            self.save_metrics_history()

    def save_metrics_history(self):
        """Save metrics history to CSV file"""
        if not self.metrics_history['time']:
            return

        df = pd.DataFrame(self.metrics_history)
        csv_file = os.path.join(self.output_dir, 'monitoring_metrics.csv')
        df.to_csv(csv_file, index=False)

    def start_monitoring(self):
        """Start monitoring the directory for new VTK files"""
        self.monitoring = True
        print(f"Starting monitoring of directory: {self.watch_dir}")

        # Process existing files first
        existing_files = sorted(glob.glob(os.path.join(self.watch_dir, "*.vtk")))
        if existing_files:
            latest_file = existing_files[-1]
            self.process_new_file(latest_file)

        # Start file system observer
        self.observer.schedule(self.file_handler, self.watch_dir, recursive=False)
        self.observer.start()

        print("Monitoring started. Close the plot window to stop.")

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        self.observer.stop()
        self.observer.join()
        print("Monitoring stopped.")

class VTKFileHandler(FileSystemEventHandler):
    """File system event handler for VTK files"""

    def __init__(self, monitor):
        self.monitor = monitor

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.vtk'):
            # Wait a moment for file to be fully written
            time.sleep(0.5)
            self.monitor.process_new_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.vtk'):
            # Check if this is a new file or just an update
            if event.src_path != self.monitor.last_processed_file:
                time.sleep(0.5)
                self.monitor.process_new_file(event.src_path)

def manual_monitoring(watch_dir, output_dir, interval=2.0):
    """Manual monitoring mode without file system events"""
    monitor = CFDMonitor(watch_dir, output_dir)

    # Show initial plot
    plt.ion()  # Turn on interactive mode
    plt.show()

    print(f"Manual monitoring mode. Checking {watch_dir} every {interval} seconds...")
    print("Press Ctrl+C to stop monitoring.")

    last_file = None
    try:
        while True:
            # Find latest VTK file
            vtk_files = glob.glob(os.path.join(watch_dir, "*.vtk"))
            if vtk_files:
                latest_file = max(vtk_files, key=os.path.getctime)

                # Process if it's a new file
                if latest_file != last_file:
                    monitor.process_new_file(latest_file)
                    last_file = latest_file

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")
        monitor.save_metrics_history()

def main():
    parser = argparse.ArgumentParser(description='Real-time CFD monitoring dashboard')
    parser.add_argument('--watch_dir', '-w', default='../../output/vtk_files',
                       help='Directory to monitor for VTK files')
    parser.add_argument('--output', '-o', default='visualization_output',
                       help='Output directory for saved data')
    parser.add_argument('--interval', '-i', type=float, default=2.0,
                       help='Monitoring interval in seconds (manual mode)')
    parser.add_argument('--manual', '-m', action='store_true',
                       help='Use manual polling instead of file system events')

    args = parser.parse_args()

    if not os.path.exists(args.watch_dir):
        print(f"Watch directory does not exist: {args.watch_dir}")
        return

    try:
        if args.manual:
            manual_monitoring(args.watch_dir, args.output, args.interval)
        else:
            # Automatic monitoring with file system events
            monitor = CFDMonitor(args.watch_dir, args.output)

            # Setup matplotlib for interactive use
            plt.ion()
            plt.show()

            monitor.start_monitoring()

            try:
                # Keep the main thread alive
                while monitor.monitoring:
                    plt.pause(1.0)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
            finally:
                monitor.stop_monitoring()

    except ImportError:
        print("Warning: watchdog package not available. Using manual mode.")
        manual_monitoring(args.watch_dir, args.output, args.interval)

    print("Real-time monitoring complete!")

if __name__ == "__main__":
    main()