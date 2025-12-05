"""
Logging utilities for HyperGraph-MAE.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict
import json
import numpy as np


def setup_logger(name: str = None, log_dir: Optional[Path] = None,
                 level: str = 'INFO', console: bool = True,
                 file: bool = True) -> logging.Logger:
    """
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file

    Returns:
        Configured logger
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers properly (avoid handler leaks)
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file and log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f"{name or 'hypergraph_mae'}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_file}")

    # Ensure module-level loggers also emit to the same handlers (trainer logs visible)
    try:
        root_logger = logging.getLogger()
        # Clean up existing root handlers properly
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        for h in logger.handlers:
            root_logger.addHandler(h)
        root_logger.setLevel(logger.level)
    except Exception:
        pass
    # Prevent double logging when this function is called multiple times
    logger.propagate = False

    return logger


class ExperimentLogger:
    """
    Logger for tracking experiments with metrics and configuration.
    """

    def __init__(self, experiment_name: str, log_dir: Path,
                 config: Optional[Dict] = None, minimal_output: bool = False):
        """
        Initialize experiment logger.

        Args:
            experiment_name: Name of the experiment
            log_dir: Directory for logs
            config: Experiment configuration
            minimal_output: If True, create simplified directory structure and fewer files
        """
        self.experiment_name = experiment_name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.minimal_output = minimal_output

        if minimal_output:
            # Minimal mode:
            # - prefer the given log_dir as run directory
            # - create a subdir if name mismatch to avoid mixing runs
            # - append timestamp suffix if directory already exists
            if experiment_name and self.log_dir.name == experiment_name:
                base_dir = self.log_dir
            else:
                base_dir = self.log_dir / (experiment_name or 'run')

            if base_dir.exists():
                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                base_dir = base_dir.parent / f"{base_dir.name}_{ts}"

            self.experiment_dir = base_dir
        else:
            # Full mode: create timestamped experiment directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.experiment_dir = self.log_dir / f"{experiment_name}_{timestamp}"
        
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # Setup logger
        self.logger = setup_logger(
            name=experiment_name,
            log_dir=self.experiment_dir,
            level='INFO'
        )

        # Save configuration
        if config:
            self.save_config(config)

        # Initialize metrics storage
        self.metrics = {
            'train': [],
            'val': [],
            'test': []
        }

    def save_config(self, config: Dict):
        """Save experiment configuration (no-op in minimal mode)."""
        if not self.minimal_output:
            config_path = self.experiment_dir / 'config.json'
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            self.logger.info(f"Configuration saved to {config_path}")
        # Skip in minimal mode; often have trial_xxx_config.yaml

    def log_metrics(self, metrics: Dict, phase: str = 'train', step: int = None):
        """
        Log metrics for a specific phase.

        Args:
            metrics: Dictionary of metrics
            phase: Training phase ('train', 'val', 'test')
            step: Current step/epoch
        """
        # Add timestamp and step
        metrics_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }

        # Store metrics - initialize phase key if it doesn't exist
        if phase not in self.metrics:
            self.metrics[phase] = []
        self.metrics[phase].append(metrics_entry)

        # Log to file (JSON only when not in minimal mode)
        if not self.minimal_output:
            metrics_file = self.experiment_dir / f'{phase}_metrics.json'
            with open(metrics_file, 'w') as f:
                json.dump(self.metrics[phase], f, indent=2)

        # Log summary
        log_str = f"{phase.capitalize()} metrics"
        if step is not None:
            log_str += f" (step {step})"
        log_str += ": " + ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()
                                     if isinstance(v, (int, float))])
        self.logger.info(log_str)

    def log_artifact(self, artifact_path: Path, artifact_type: str = 'model'):
        """
        Log an artifact (model, plot, etc.).

        Args:
            artifact_path: Path to artifact
            artifact_type: Type of artifact
        """
        # Create artifacts directory
        artifacts_dir = self.experiment_dir / 'artifacts'
        artifacts_dir.mkdir(exist_ok=True)

        # Copy artifact
        import shutil
        dest_path = artifacts_dir / f"{artifact_type}_{artifact_path.name}"
        shutil.copy2(artifact_path, dest_path)

        self.logger.info(f"{artifact_type.capitalize()} saved to {dest_path}")

    def log_text(self, text: str, filename: str = 'notes.txt'):
        """
        Log text to a file with proper error handling.

        Args:
            text: Text to log
            filename: Output filename
        """
        text_path = self.experiment_dir / filename
        try:
            with open(text_path, 'a', encoding='utf-8') as f:
                f.write(f"\n[{datetime.now().isoformat()}]\n")
                f.write(text)
                f.write("\n")
                f.flush()  # Ensure data is written immediately
        except OSError as e:
            # Fallback if file operations fail (prevent crash from file handle exhaustion)
            print(f"Warning: Could not write to {text_path}: {e}")

    def get_summary(self) -> Dict:
        """
        Get experiment summary.

        Returns:
            Dictionary with experiment summary
        """
        summary = {
            'experiment_name': self.experiment_name,
            'experiment_dir': str(self.experiment_dir),
            'start_time': self.metrics['train'][0]['timestamp'] if self.metrics['train'] else None,
            'end_time': self.metrics['train'][-1]['timestamp'] if self.metrics['train'] else None,
            'num_epochs': len(self.metrics['train']),
            'best_metrics': {}
        }

        # Find best metrics
        for phase in ['train', 'val', 'test']:
            if self.metrics[phase]:
                # Extract numeric metrics
                numeric_metrics = {}
                for entry in self.metrics[phase]:
                    for k, v in entry.items():
                        if isinstance(v, (int, float)) and k not in ['step', 'timestamp']:
                            if k not in numeric_metrics:
                                numeric_metrics[k] = []
                            numeric_metrics[k].append(v)

                # Find best values
                best = {}
                for k, values in numeric_metrics.items():
                    if 'loss' in k.lower():
                        best[f'min_{k}'] = min(values)
                    else:
                        best[f'max_{k}'] = max(values)

                summary['best_metrics'][phase] = best

        return summary

    def save_summary(self):
        """Save experiment summary (no-op in minimal mode)."""
        if not self.minimal_output:
            summary = self.get_summary()
            summary_path = self.experiment_dir / 'summary.json'

            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)

            self.logger.info(f"Experiment summary saved to {summary_path}")
        # Skip summary saving in minimal mode


class ProgressLogger:
    """
    Logger for tracking training progress with ETA estimation.
    """

    def __init__(self, total_steps: int, log_interval: int = 10):
        """
        Initialize progress logger.

        Args:
            total_steps: Total number of steps
            log_interval: Logging interval
        """
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.current_step = 0
        self.start_time = datetime.now()
        self.step_times = []

    def update(self, step: int = None, metrics: Optional[Dict] = None):
        """
        Update progress.

        Args:
            step: Current step (if None, increment by 1)
            metrics: Optional metrics to log
        """
        if step is not None:
            self.current_step = step
        else:
            self.current_step += 1

        # Track step time
        current_time = datetime.now()
        if len(self.step_times) > 0:
            step_duration = (current_time - self.step_times[-1][1]).total_seconds()
            self.step_times.append((self.current_step, current_time, step_duration))
        else:
            self.step_times.append((self.current_step, current_time, 0))

        # Keep only recent times for ETA calculation
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]

        # Log progress
        if self.current_step % self.log_interval == 0:
            self._log_progress(metrics)

    def _log_progress(self, metrics: Optional[Dict] = None):
        """Log current progress with ETA."""
        # Calculate progress
        progress = self.current_step / self.total_steps * 100

        # Calculate ETA
        if len(self.step_times) > 1:
            recent_steps = self.step_times[-min(20, len(self.step_times)):]
            avg_step_time = np.mean([t[2] for t in recent_steps[1:]])
            remaining_steps = self.total_steps - self.current_step
            eta_seconds = avg_step_time * remaining_steps
            eta = datetime.now() + timedelta(seconds=eta_seconds)
            eta_str = eta.strftime('%Y-%m-%d %H:%M:%S')
        else:
            eta_str = 'Unknown'

        # Build log message
        log_msg = f"Step {self.current_step}/{self.total_steps} ({progress:.1f}%)"
        log_msg += f" | ETA: {eta_str}"

        if metrics:
            metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()
                                      if isinstance(v, (int, float))])
            log_msg += f" | {metrics_str}"

        logging.info(log_msg)

    def finish(self):
        """Log completion message."""
        total_time = (datetime.now() - self.start_time).total_seconds()
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)

        logging.info(f"Training completed in {hours}h {minutes}m {seconds}s")


from datetime import timedelta
