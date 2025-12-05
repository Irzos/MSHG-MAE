"""
Memory management utilities for efficient training.
"""

import torch
import gc
import os
import psutil
import logging
from typing import Dict, Optional
import GPUtil

logger = logging.getLogger(__name__)

# Global process instance to avoid creating multiple Process objects
_process_instance = None

def _get_process():
    """Get singleton process instance to prevent resource leaks."""
    global _process_instance
    if _process_instance is None:
        _process_instance = psutil.Process(os.getpid())
    return _process_instance


def get_memory_info() -> Dict:
    """
    Get current memory usage information.

    Returns:
        Dictionary with memory statistics
    """
    info = {}

    # CPU memory  
    process = _get_process()
    cpu_memory = process.memory_info()
    info['cpu'] = {
        'used_gb': cpu_memory.rss / (1024 ** 3),
        'percent': process.memory_percent(),
        'available_gb': psutil.virtual_memory().available / (1024 ** 3)
    }

    # GPU memory
    if torch.cuda.is_available():
        info['gpu'] = {}

        # PyTorch CUDA memory
        for i in range(torch.cuda.device_count()):
            info['gpu'][f'cuda:{i}'] = {
                'allocated_gb': torch.cuda.memory_allocated(i) / (1024 ** 3),
                'reserved_gb': torch.cuda.memory_reserved(i) / (1024 ** 3),
                'free_gb': (torch.cuda.get_device_properties(i).total_memory -
                            torch.cuda.memory_allocated(i)) / (1024 ** 3)
            }

        # GPUtil for additional GPU info
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                info['gpu'][f'cuda:{i}'].update({
                    'utilization': gpu.load * 100,
                    'temperature': gpu.temperature
                })
        except:
            pass

    return info


def log_memory_usage(prefix: str = ""):
    """
    Log current memory usage.

    Args:
        prefix: Prefix for log message
    """
    memory_info = get_memory_info()

    # CPU memory
    cpu_info = memory_info['cpu']
    log_msg = f"{prefix}CPU Memory: {cpu_info['used_gb']:.2f}GB ({cpu_info['percent']:.1f}%)"

    # GPU memory
    if 'gpu' in memory_info:
        for device, gpu_info in memory_info['gpu'].items():
            log_msg += f" | {device}: {gpu_info['allocated_gb']:.2f}GB allocated"
            if 'utilization' in gpu_info:
                log_msg += f" ({gpu_info['utilization']:.1f}% util)"

    logger.info(log_msg)


def cleanup_memory():
    """
    Clean up memory by running garbage collection and clearing GPU cache.
    """
    # Python garbage collection
    gc.collect()

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def optimize_memory_allocation():
    """
    Optimize PyTorch memory allocation settings.
    """
    if torch.cuda.is_available():
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.9)

        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True

        # Set allocation configuration
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8'

        logger.info("Memory allocation optimized for CUDA")


class MemoryMonitor:
    """
    Monitor memory usage during training.
    """

    def __init__(self, log_interval: int = 100, alert_threshold: float = 0.9):
        """
        Initialize memory monitor.

        Args:
            log_interval: Interval for logging memory stats
            alert_threshold: Memory usage threshold for alerts
        """
        self.log_interval = log_interval
        self.alert_threshold = alert_threshold
        self.step_count = 0
        self.memory_history = []

    def check(self):
        """Check memory usage and log if needed."""
        self.step_count += 1

        if self.step_count % self.log_interval == 0:
            memory_info = get_memory_info()
            self.memory_history.append(memory_info)

            # Check for high memory usage
            self._check_alerts(memory_info)

            # Log current usage
            log_memory_usage(f"Step {self.step_count}: ")

    def _check_alerts(self, memory_info: Dict):
        """Check for memory usage alerts."""
        # CPU memory alert
        if memory_info['cpu']['percent'] > self.alert_threshold * 100:
            logger.warning(f"High CPU memory usage: {memory_info['cpu']['percent']:.1f}%")

        # GPU memory alert
        if 'gpu' in memory_info:
            for device, gpu_info in memory_info['gpu'].items():
                usage = gpu_info['allocated_gb'] / (gpu_info['allocated_gb'] + gpu_info['free_gb'])
                if usage > self.alert_threshold:
                    logger.warning(f"High GPU memory usage on {device}: {usage * 100:.1f}%")

    def get_peak_usage(self) -> Dict:
        """Get peak memory usage from history."""
        if not self.memory_history:
            return {}

        peak = {
            'cpu': {'used_gb': 0, 'percent': 0},
            'gpu': {}
        }

        for entry in self.memory_history:
            # CPU peak
            if entry['cpu']['used_gb'] > peak['cpu']['used_gb']:
                peak['cpu'] = entry['cpu']

            # GPU peak
            if 'gpu' in entry:
                for device, gpu_info in entry['gpu'].items():
                    if device not in peak['gpu']:
                        peak['gpu'][device] = {'allocated_gb': 0}
                    if gpu_info['allocated_gb'] > peak['gpu'][device]['allocated_gb']:
                        peak['gpu'][device] = gpu_info

        return peak


def estimate_batch_size(model: torch.nn.Module, input_shape: tuple,
                        target_memory_usage: float = 0.8) -> int:
    """
    Estimate optimal batch size based on available memory.

    Args:
        model: Model to test
        input_shape: Shape of single input (without batch dimension)
        target_memory_usage: Target GPU memory usage (0-1)

    Returns:
        Estimated optimal batch size
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, returning default batch size")
        return 32

    device = next(model.parameters()).device

    # Get available memory
    total_memory = torch.cuda.get_device_properties(device).total_memory
    allocated_memory = torch.cuda.memory_allocated(device)
    available_memory = (total_memory - allocated_memory) * target_memory_usage

    # Test with small batch
    test_batch_size = 4
    cleanup_memory()

    try:
        # Create test input
        test_input = torch.randn(test_batch_size, *input_shape, device=device)

        # Forward pass
        start_memory = torch.cuda.memory_allocated(device)
        with torch.no_grad():
            _ = model(test_input)
        torch.cuda.synchronize()

        # Calculate memory per sample
        memory_per_sample = (torch.cuda.memory_allocated(device) - start_memory) / test_batch_size

        # Estimate batch size (with safety margin)
        estimated_batch_size = int(available_memory / memory_per_sample * 0.9)

        # Ensure reasonable bounds
        estimated_batch_size = max(1, min(estimated_batch_size, 1024))

        logger.info(f"Estimated optimal batch size: {estimated_batch_size}")
        return estimated_batch_size

    except Exception as e:
        logger.error(f"Error estimating batch size: {e}")
        return 32
    finally:
        cleanup_memory()


class MemoryEfficientDataLoader:
    """
    Memory-efficient data loader with prefetching and pinned memory.
    """

    def __init__(self, dataset, batch_size: int, num_workers: int = 4,
                 pin_memory: bool = True, prefetch_factor: int = 2):
        """
        Initialize memory-efficient data loader.

        Args:
            dataset: Dataset to load
            batch_size: Batch size
            num_workers: Number of worker processes
            pin_memory: Whether to use pinned memory
            prefetch_factor: Number of batches to prefetch
        """
        self.loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            prefetch_factor=prefetch_factor if num_workers > 0 else 2,
            persistent_workers=num_workers > 0,
            drop_last=True
        )

    def __iter__(self):
        """Iterate with memory monitoring."""
        monitor = MemoryMonitor(log_interval=50)

        for batch in self.loader:
            monitor.check()
            yield batch

        # Log peak usage
        peak = monitor.get_peak_usage()
        logger.info(f"Peak memory usage - CPU: {peak['cpu']['used_gb']:.2f}GB")
        if 'gpu' in peak:
            for device, info in peak['gpu'].items():
                logger.info(f"Peak memory usage - {device}: {info['allocated_gb']:.2f}GB")


def profile_memory_usage(func):
    """
    Decorator to profile memory usage of a function.

    Args:
        func: Function to profile

    Returns:
        Wrapped function with memory profiling
    """

    def wrapper(*args, **kwargs):
        # Initial memory state
        cleanup_memory()
        initial_cpu = _get_process().memory_info().rss / (1024 ** 3)
        initial_gpu = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                initial_gpu[i] = torch.cuda.memory_allocated(i) / (1024 ** 3)

        # Run function
        result = func(*args, **kwargs)

        # Final memory state
        final_cpu = _get_process().memory_info().rss / (1024 ** 3)
        final_gpu = {}

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                final_gpu[i] = torch.cuda.memory_allocated(i) / (1024 ** 3)

        # Log memory usage
        logger.info(f"Memory usage for {func.__name__}:")
        logger.info(f"  CPU: {initial_cpu:.2f}GB -> {final_cpu:.2f}GB (Δ {final_cpu - initial_cpu:.2f}GB)")

        for i in initial_gpu:
            logger.info(f"  GPU {i}: {initial_gpu[i]:.2f}GB -> {final_gpu[i]:.2f}GB "
                        f"(Δ {final_gpu[i] - initial_gpu[i]:.2f}GB)")

        return result

    return wrapper
