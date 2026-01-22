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

import argparse
import glob
import os
import time
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from cfd_viz.analysis.time_series import (
    FlowMetrics,
    compute_flow_metrics,
    create_flow_metrics_time_series,
)
from cfd_viz.common import (
    DATA_DIR,
    PLOTS_DIR,
    ensure_dirs,
    read_vtk_file as _read_vtk_file,
)


def read_vtk_file(filename: str) -> Optional[Dict[str, Any]]:
    """Read a VTK structured points file and extract data.

    Args:
        filename: Path to the VTK file.

    Returns:
        Dictionary with keys: x, y, data_fields, nx, ny, dx, dy, filename, timestamp
        or None if the file cannot be read.
    """
    try:
        data = _read_vtk_file(filename)
    except (FileNotFoundError, PermissionError):
        return None

    if data is None:
        return None

    # Map field names for backward compatibility
    data_fields = {}
    field_mapping = {
        "u": "u_velocity",
        "v": "v_velocity",
        "p": "pressure",
        "rho": "density",
        "T": "temperature",
    }
    for old_name, new_name in field_mapping.items():
        if old_name in data.fields:
            data_fields[new_name] = data.fields[old_name]

    # Also include any fields with their original names
    for name, field in data.fields.items():
        if name not in field_mapping:
            data_fields[name] = field

    if "u_velocity" not in data_fields or "v_velocity" not in data_fields:
        return None

    return {
        "x": data.x,
        "y": data.y,
        "data_fields": data_fields,
        "nx": data.nx,
        "ny": data.ny,
        "dx": data.dx,
        "dy": data.dy,
        "filename": filename,
        "timestamp": time.time(),
    }


class CFDMonitor:
    def __init__(self, watch_dir, output_dir="visualization_output"):
        self.watch_dir = watch_dir
        self.output_dir = output_dir
        self.monitoring = False
        self.data_history = []
        self.time_history = []
        # Use FlowMetricsTimeSeries from analysis module
        self.monitoring_history = create_flow_metrics_time_series(max_length=100)

        # Setup plotting
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle("CFD Real-time Monitoring Dashboard", fontsize=16)
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
        self.ax_velocity.set_title("Latest Velocity Field")
        self.ax_velocity.set_xlabel("x (m)")
        self.ax_velocity.set_ylabel("y (m)")

        # 2. Pressure field (latest)
        self.ax_pressure = self.axes[1]
        self.ax_pressure.set_title("Latest Pressure Field")
        self.ax_pressure.set_xlabel("x (m)")
        self.ax_pressure.set_ylabel("y (m)")

        # 3. Max velocity vs time
        self.ax_max_vel = self.axes[2]
        self.ax_max_vel.set_title("Maximum Velocity vs Time")
        self.ax_max_vel.set_xlabel("Time Step")
        self.ax_max_vel.set_ylabel("Max Velocity (m/s)")
        self.ax_max_vel.grid(True, alpha=0.3)

        # 4. Mean velocity vs time
        self.ax_mean_vel = self.axes[3]
        self.ax_mean_vel.set_title("Mean Velocity vs Time")
        self.ax_mean_vel.set_xlabel("Time Step")
        self.ax_mean_vel.set_ylabel("Mean Velocity (m/s)")
        self.ax_mean_vel.grid(True, alpha=0.3)

        # 5. Kinetic energy vs time
        self.ax_energy = self.axes[4]
        self.ax_energy.set_title("Total Kinetic Energy vs Time")
        self.ax_energy.set_xlabel("Time Step")
        self.ax_energy.set_ylabel("Kinetic Energy")
        self.ax_energy.grid(True, alpha=0.3)

        # 6. Statistics summary
        self.ax_stats = self.axes[5]
        self.ax_stats.set_title("Current Statistics")
        self.ax_stats.axis("off")

        plt.tight_layout()

    def calculate_metrics(self, data) -> FlowMetrics:
        """Calculate flow metrics for monitoring using analysis module.

        Returns:
            FlowMetrics with computed flow values.
        """
        fields = data["data_fields"]
        u = fields["u_velocity"]
        v = fields["v_velocity"]
        p = fields.get("pressure", np.zeros_like(u))

        # Use the pure function from analysis module
        metrics = compute_flow_metrics(
            u=u,
            v=v,
            p=p,
            dx=data["dx"],
            dy=data["dy"],
            timestamp=data["timestamp"],
        )

        return metrics

    def update_plots(self, data):
        """Update all monitoring plots with new data"""
        if data is None:
            return

        # Calculate metrics using the analysis module
        snapshot = self.calculate_metrics(data)

        # Add to monitoring history (handles max_length internally)
        self.monitoring_history.add(snapshot)

        # Get current data for plotting
        fields = data["data_fields"]
        u = fields["u_velocity"]
        v = fields["v_velocity"]
        pressure = fields.get("pressure", np.zeros_like(u))
        X, Y = np.meshgrid(data["x"], data["y"])
        velocity_mag = np.sqrt(u**2 + v**2)

        # Update velocity field
        self.ax_velocity.clear()
        self.ax_velocity.contourf(X, Y, velocity_mag, levels=20, cmap="viridis")
        self.ax_velocity.set_title(
            f"Velocity Field - {os.path.basename(data['filename'])}"
        )
        self.ax_velocity.set_xlabel("x (m)")
        self.ax_velocity.set_ylabel("y (m)")
        self.ax_velocity.set_aspect("equal")

        # Update pressure field
        self.ax_pressure.clear()
        self.ax_pressure.contourf(X, Y, pressure, levels=20, cmap="plasma")
        self.ax_pressure.set_title(
            f"Pressure Field - {os.path.basename(data['filename'])}"
        )
        self.ax_pressure.set_xlabel("x (m)")
        self.ax_pressure.set_ylabel("y (m)")
        self.ax_pressure.set_aspect("equal")

        # Update time series plots using FlowMetricsTimeSeries methods
        num_snapshots = len(self.monitoring_history.snapshots)
        if num_snapshots > 1:
            time_steps = range(num_snapshots)

            # Get metric arrays from history
            max_velocity = self.monitoring_history.get_metric_array("max_velocity")
            mean_velocity = self.monitoring_history.get_metric_array("mean_velocity")
            total_ke = self.monitoring_history.get_metric_array("total_kinetic_energy")

            # Max velocity
            self.ax_max_vel.clear()
            self.ax_max_vel.plot(time_steps, max_velocity, "b-o", markersize=4)
            self.ax_max_vel.set_title("Maximum Velocity vs Time")
            self.ax_max_vel.set_xlabel("Time Step")
            self.ax_max_vel.set_ylabel("Max Velocity (m/s)")
            self.ax_max_vel.grid(True, alpha=0.3)

            # Mean velocity
            self.ax_mean_vel.clear()
            self.ax_mean_vel.plot(time_steps, mean_velocity, "r-o", markersize=4)
            self.ax_mean_vel.set_title("Mean Velocity vs Time")
            self.ax_mean_vel.set_xlabel("Time Step")
            self.ax_mean_vel.set_ylabel("Mean Velocity (m/s)")
            self.ax_mean_vel.grid(True, alpha=0.3)

            # Kinetic energy
            self.ax_energy.clear()
            self.ax_energy.plot(time_steps, total_ke, "g-o", markersize=4)
            self.ax_energy.set_title("Total Kinetic Energy vs Time")
            self.ax_energy.set_xlabel("Time Step")
            self.ax_energy.set_ylabel("Kinetic Energy")
            self.ax_energy.grid(True, alpha=0.3)

        # Update statistics
        self.ax_stats.clear()
        self.ax_stats.axis("off")

        stats_text = f"""
        Current Flow Statistics:

        Velocity:
        Max: {snapshot.max_velocity:.4f} m/s
        Mean: {snapshot.mean_velocity:.4f} m/s

        Pressure:
        Max: {snapshot.max_pressure:.4f}
        Mean: {snapshot.mean_pressure:.4f}

        Energy:
        Total KE: {snapshot.total_kinetic_energy:.4f}

        Vorticity:
        Max |ω|: {snapshot.max_vorticity:.4f} 1/s

        Grid: {data["nx"]} x {data["ny"]}
        File: {os.path.basename(data["filename"])}
        Time: {time.strftime("%H:%M:%S", time.localtime(snapshot.timestamp))}

        Monitoring Status: {"Active" if self.monitoring else "Stopped"}
        Files Processed: {len(self.monitoring_history.snapshots)}
        """

        self.ax_stats.text(
            0.05,
            0.95,
            stats_text,
            transform=self.ax_stats.transAxes,
            fontsize=10,
            verticalalignment="top",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )

        # Calculate convergence trends using FlowMetricsTimeSeries methods
        if len(self.monitoring_history.snapshots) > 5:
            max_vel_trend = self.monitoring_history.estimate_convergence_trend(
                "max_velocity", window=5
            )
            mean_vel_trend = self.monitoring_history.estimate_convergence_trend(
                "mean_velocity", window=5
            )
            is_converged = self.monitoring_history.is_converged(threshold=1e-5)

            convergence_text = f"""
            Convergence Analysis:
            Max Velocity Trend: {max_vel_trend:+.6f}/step
            Mean Velocity Trend: {mean_vel_trend:+.6f}/step

            Status: {"Converging" if is_converged else "Still Changing"}
            """

            self.ax_stats.text(
                0.05,
                0.45,
                convergence_text,
                transform=self.ax_stats.transAxes,
                fontsize=9,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
            )

        plt.tight_layout()
        plt.draw()

    def process_new_file(self, filepath):
        """Process a new VTK file"""
        print(f"Processing new file: {os.path.basename(filepath)}")

        data = read_vtk_file(filepath)
        if data:
            self.update_plots(data)
            self.last_processed_file = filepath

            # Save metrics to CSV
            self.save_metrics_history()

    def save_metrics_history(self):
        """Save metrics history to CSV file"""
        if not self.monitoring_history.snapshots:
            return

        # Convert FlowMetricsTimeSeries to DataFrame
        metrics_dict = {
            "timestamp": self.monitoring_history.get_metric_array("timestamp"),
            "max_velocity": self.monitoring_history.get_metric_array("max_velocity"),
            "mean_velocity": self.monitoring_history.get_metric_array("mean_velocity"),
            "max_pressure": self.monitoring_history.get_metric_array("max_pressure"),
            "mean_pressure": self.monitoring_history.get_metric_array("mean_pressure"),
            "total_kinetic_energy": self.monitoring_history.get_metric_array(
                "total_kinetic_energy"
            ),
            "max_vorticity": self.monitoring_history.get_metric_array("max_vorticity"),
        }
        df = pd.DataFrame(metrics_dict)
        csv_file = os.path.join(self.output_dir, "monitoring_metrics.csv")
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
        if not event.is_directory and event.src_path.endswith(".vtk"):
            # Wait a moment for file to be fully written
            time.sleep(0.5)
            self.monitor.process_new_file(event.src_path)

    def on_modified(self, event):
        if (
            not event.is_directory
            and event.src_path.endswith(".vtk")
            and event.src_path != self.monitor.last_processed_file
        ):
            # Check if this is a new file or just an update
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
    # Ensure output directories exist
    ensure_dirs()

    parser = argparse.ArgumentParser(description="Real-time CFD monitoring dashboard")
    parser.add_argument(
        "--watch_dir",
        "-w",
        default=None,
        help="Directory to monitor for VTK files (default: centralized DATA_DIR)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for saved data (default: centralized PLOTS_DIR)",
    )
    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=2.0,
        help="Monitoring interval in seconds (manual mode)",
    )
    parser.add_argument(
        "--manual",
        "-m",
        action="store_true",
        help="Use manual polling instead of file system events",
    )

    args = parser.parse_args()

    # Use centralized directories if not specified
    watch_dir = args.watch_dir if args.watch_dir else str(DATA_DIR)
    output_dir = args.output if args.output else str(PLOTS_DIR)

    if not os.path.exists(watch_dir):
        print(f"Watch directory does not exist: {watch_dir}")
        print("Set CFD_VIZ_DATA_DIR environment variable to specify data location.")
        return

    try:
        if args.manual:
            manual_monitoring(watch_dir, output_dir, args.interval)
        else:
            # Automatic monitoring with file system events
            monitor = CFDMonitor(watch_dir, output_dir)

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
        manual_monitoring(watch_dir, output_dir, args.interval)

    print("Real-time monitoring complete!")


if __name__ == "__main__":
    main()
