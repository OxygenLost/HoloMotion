#!/usr/bin/env python3

"""
Comprehensive Rollout Profiler for IsaacLab PPO Training

This script provides detailed profiling tools to identify performance bottlenecks
in the rollout collection phase of PPO training with IsaacLab environments.

Usage:
1. Import this module in your PPO code
2. Create a RolloutProfiler instance 
3. Wrap critical sections with profiler.start_timing()/end_timing()
4. Call profiler.print_results() to see the breakdown

The profiler measures:
- Environment stepping time
- Actor inference time  
- Critic inference time
- Device transfers
- Storage operations
- Memory allocations
"""

import time
import torch
import psutil
import gc
from collections import defaultdict, deque
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger


class RolloutProfiler:
    """Comprehensive profiler for PPO rollout operations."""
    
    def __init__(self, 
                 max_samples: int = 100,
                 track_memory: bool = True,
                 track_cuda_events: bool = True):
        self.max_samples = max_samples
        self.track_memory = track_memory
        self.track_cuda_events = track_cuda_events
        
        # Timing storage
        self.timings: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.current_timers: Dict[str, float] = {}
        
        # Memory tracking
        self.memory_stats: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        
        # CUDA events for more precise GPU timing
        if track_cuda_events and torch.cuda.is_available():
            self.cuda_events: Dict[str, List[torch.cuda.Event]] = defaultdict(list)
            self.cuda_timers: Dict[str, torch.cuda.Event] = {}
        else:
            self.cuda_events = {}
            self.cuda_timers = {}
            
        # Step counters
        self.step_count = 0
        self.enabled = True
        
    def enable(self):
        """Enable profiling."""
        self.enabled = True
        
    def disable(self):
        """Disable profiling to avoid overhead."""
        self.enabled = False
        
    def start_timing(self, operation: str):
        """Start timing an operation."""
        if not self.enabled:
            return
            
        # CPU timing
        self.current_timers[operation] = time.perf_counter()
        
        # GPU timing
        if operation not in self.cuda_events and torch.cuda.is_available():
            self.cuda_events[operation] = []
        
        if torch.cuda.is_available() and self.track_cuda_events:
            torch.cuda.synchronize()  # Ensure all GPU operations complete
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            self.cuda_timers[operation] = start_event
            
        # Memory tracking
        if self.track_memory:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear cache for accurate measurement
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                gpu_memory = 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            self.memory_stats[f"{operation}_start"] = {
                'cpu_memory': cpu_memory,
                'gpu_memory': gpu_memory
            }
    
    def end_timing(self, operation: str):
        """End timing an operation."""
        if not self.enabled:
            return
            
        # CPU timing
        if operation in self.current_timers:
            elapsed = time.perf_counter() - self.current_timers[operation]
            self.timings[operation].append(elapsed)
            del self.current_timers[operation]
        
        # GPU timing
        if torch.cuda.is_available() and self.track_cuda_events and operation in self.cuda_timers:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            torch.cuda.synchronize()  # Wait for all operations to complete
            
            start_event = self.cuda_timers[operation]
            gpu_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
            self.timings[f"{operation}_gpu"].append(gpu_time)
            
            del self.cuda_timers[operation]
            
        # Memory tracking
        if self.track_memory:
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1024**2  # MB
            else:
                gpu_memory = 0
            cpu_memory = psutil.Process().memory_info().rss / 1024**2  # MB
            
            start_stats = self.memory_stats.get(f"{operation}_start", {})
            if start_stats:
                self.memory_stats[f"{operation}_memory_delta"].append({
                    'cpu_delta': cpu_memory - start_stats.get('cpu_memory', 0),
                    'gpu_delta': gpu_memory - start_stats.get('gpu_memory', 0)
                })
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for an operation."""
        if operation not in self.timings or len(self.timings[operation]) == 0:
            return {}
            
        times = list(self.timings[operation])
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
            'p95': np.percentile(times, 95),
            'count': len(times)
        }
    
    def print_results(self, top_n: int = 10):
        """Print comprehensive profiling results."""
        if not self.timings:
            logger.warning("No profiling data collected!")
            return
            
        logger.info("=" * 80)
        logger.info("ROLLOUT PERFORMANCE PROFILING RESULTS")
        logger.info("=" * 80)
        
        # Sort operations by mean time
        operation_stats = {}
        for op in self.timings.keys():
            stats = self.get_stats(op)
            if stats:
                operation_stats[op] = stats
                
        sorted_ops = sorted(operation_stats.items(), 
                          key=lambda x: x[1]['mean'], reverse=True)[:top_n]
        
        # Print timing results
        logger.info("\n📊 TIMING BREAKDOWN (Top {} slowest operations):".format(top_n))
        logger.info("-" * 80)
        logger.info(f"{'Operation':<25} {'Mean(ms)':<10} {'Std(ms)':<10} {'P95(ms)':<10} {'Count':<8}")
        logger.info("-" * 80)
        
        total_time = 0
        for op, stats in sorted_ops:
            mean_ms = stats['mean'] * 1000
            std_ms = stats['std'] * 1000
            p95_ms = stats['p95'] * 1000
            count = stats['count']
            total_time += stats['mean'] * count
            
            logger.info(f"{op:<25} {mean_ms:<10.2f} {std_ms:<10.2f} {p95_ms:<10.2f} {count:<8}")
        
        # Print memory usage if available
        if self.track_memory and any('_memory_delta' in k for k in self.memory_stats.keys()):
            logger.info("\n🧠 MEMORY USAGE ANALYSIS:")
            logger.info("-" * 80)
            logger.info(f"{'Operation':<25} {'CPU Δ(MB)':<12} {'GPU Δ(MB)':<12}")
            logger.info("-" * 80)
            
            for op, stats in sorted_ops:
                mem_key = f"{op}_memory_delta"
                if mem_key in self.memory_stats and len(self.memory_stats[mem_key]) > 0:
                    mem_deltas = list(self.memory_stats[mem_key])
                    avg_cpu_delta = np.mean([d['cpu_delta'] for d in mem_deltas])
                    avg_gpu_delta = np.mean([d['gpu_delta'] for d in mem_deltas])
                    logger.info(f"{op:<25} {avg_cpu_delta:<12.2f} {avg_gpu_delta:<12.2f}")
        
        # Performance recommendations
        logger.info("\n🚀 PERFORMANCE RECOMMENDATIONS:")
        logger.info("-" * 80)
        self._print_recommendations(sorted_ops)
        
        logger.info("=" * 80)
    
    def _print_recommendations(self, sorted_ops: List):
        """Print performance optimization recommendations."""
        for op, stats in sorted_ops[:5]:  # Top 5 slowest operations
            mean_ms = stats['mean'] * 1000
            
            if 'device_transfer' in op.lower():
                logger.info(f"⚠️  High device transfer time ({mean_ms:.1f}ms): Consider batching transfers or keeping data on GPU")
            elif 'env_step' in op.lower():
                logger.info(f"⚠️  Slow environment stepping ({mean_ms:.1f}ms): Check simulation settings and physics parameters")
            elif 'actor' in op.lower() or 'critic' in op.lower():
                logger.info(f"⚠️  Slow neural network inference ({mean_ms:.1f}ms): Consider model optimization or batch processing")
            elif 'storage' in op.lower():
                logger.info(f"⚠️  Slow storage operations ({mean_ms:.1f}ms): Check tensor copying and data structures")
            elif 'compute_returns' in op.lower():
                logger.info(f"⚠️  Slow return computation ({mean_ms:.1f}ms): Consider vectorized operations")
    
    def reset_stats(self):
        """Reset all profiling statistics."""
        self.timings.clear()
        self.memory_stats.clear()
        self.cuda_events.clear()
        self.step_count = 0
        
    def context_timer(self, operation: str):
        """Context manager for timing operations."""
        return ProfilerContext(self, operation)


class ProfilerContext:
    """Context manager for automatic timing."""
    
    def __init__(self, profiler: RolloutProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
    
    def __enter__(self):
        self.profiler.start_timing(self.operation)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_timing(self.operation)


def analyze_device_transfers(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    """Analyze device transfer patterns in tensor dictionaries."""
    analysis = {
        'total_tensors': len(tensor_dict),
        'devices': defaultdict(int),
        'total_memory': 0,
        'transfer_needed': []
    }
    
    for key, tensor in tensor_dict.items():
        if isinstance(tensor, torch.Tensor):
            device = str(tensor.device)
            analysis['devices'][device] += 1
            analysis['total_memory'] += tensor.element_size() * tensor.numel()
            
            if device == 'cpu':
                analysis['transfer_needed'].append(key)
    
    return analysis


def create_isaac_lab_timer():
    """Create IsaacLab Timer instance if available."""
    try:
        from isaaclab.utils.timer import Timer
        return Timer()
    except ImportError:
        logger.warning("IsaacLab Timer not available, falling back to Python timer")
        return None


# Example usage function
def profile_rollout_example():
    """Example of how to use the profiler in PPO rollout."""
    profiler = RolloutProfiler(max_samples=50, track_memory=True)
    
    # Example profiling structure for rollout
    for step in range(1000):  # Simulate rollout steps
        
        with profiler.context_timer("total_rollout_step"):
            
            # Simulate actor inference
            with profiler.context_timer("actor_inference"):
                time.sleep(0.001)  # Simulate computation
                
            # Simulate critic inference  
            with profiler.context_timer("critic_inference"):
                time.sleep(0.001)
                
            # Simulate environment step
            with profiler.context_timer("env_step"):
                time.sleep(0.005)  # Environment stepping typically slower
                
            # Simulate device transfers
            with profiler.context_timer("device_transfer"):
                time.sleep(0.0005)
                
            # Simulate storage operations
            with profiler.context_timer("storage_update"):
                time.sleep(0.0002)
    
    profiler.print_results()


if __name__ == "__main__":
    profile_rollout_example()


