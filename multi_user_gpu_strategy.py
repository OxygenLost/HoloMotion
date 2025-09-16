#!/usr/bin/env python3

"""
Multi-User GPU Training Strategy for IsaacLab PPO

Optimizations for shared GPU environments where multiple users
are competing for GPU resources.
"""

import subprocess
import time
import torch
from typing import Dict, List, Tuple, Optional


def get_gpu_utilization() -> List[Dict]:
    """Get current GPU utilization for all GPUs."""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                parts = [x.strip() for x in line.split(',')]
                gpus.append({
                    'index': int(parts[0]),
                    'name': parts[1],
                    'memory_used': int(parts[2]),
                    'memory_total': int(parts[3]), 
                    'utilization': int(parts[4]),
                    'power': float(parts[5])
                })
        return gpus
    except:
        return []


def find_best_gpu() -> Optional[int]:
    """Find the best GPU for training based on current load."""
    gpus = get_gpu_utilization()
    if not gpus:
        return None
    
    # Score GPUs based on available memory and low utilization
    best_gpu = None
    best_score = -1
    
    for gpu in gpus:
        memory_free = gpu['memory_total'] - gpu['memory_used']
        memory_free_pct = memory_free / gpu['memory_total'] * 100
        utilization = gpu['utilization']
        
        # Score: prioritize free memory and low utilization
        score = memory_free_pct * 0.6 + (100 - utilization) * 0.4
        
        print(f"GPU {gpu['index']}: {memory_free_pct:.1f}% free memory, {utilization}% busy, score: {score:.1f}")
        
        if score > best_score:
            best_score = score
            best_gpu = gpu['index']
    
    return best_gpu


def wait_for_gpu_availability(target_memory_gb: float = 16.0, target_util: int = 50, 
                             timeout_minutes: int = 60) -> Optional[int]:
    """Wait for a GPU to become available with specified requirements."""
    print(f"🔍 Waiting for GPU with >{target_memory_gb}GB free and <{target_util}% utilization...")
    
    start_time = time.time()
    timeout_seconds = timeout_minutes * 60
    
    while time.time() - start_time < timeout_seconds:
        gpus = get_gpu_utilization()
        
        for gpu in gpus:
            memory_free_gb = (gpu['memory_total'] - gpu['memory_used']) / 1024
            if memory_free_gb >= target_memory_gb and gpu['utilization'] <= target_util:
                print(f"✅ Found available GPU {gpu['index']} ({memory_free_gb:.1f}GB free, {gpu['utilization']}% busy)")
                return gpu['index']
        
        print(f"⏳ Waiting... (elapsed: {(time.time() - start_time)/60:.1f} min)")
        time.sleep(30)  # Check every 30 seconds
    
    print("⚠️ Timeout reached, no GPU became available")
    return None


def optimize_for_shared_gpu(ppo_instance, smaller_batch: bool = True):
    """Apply optimizations specifically for shared GPU environment."""
    print("🚀 Applying multi-user GPU optimizations...")
    
    optimizations = []
    
    # 1. Reduce memory pressure
    if smaller_batch:
        if hasattr(ppo_instance, 'num_envs'):
            original_envs = ppo_instance.num_envs
            suggested_envs = min(original_envs, 512)  # Limit environments
            if suggested_envs < original_envs:
                optimizations.append(f"Suggest reducing num_envs: {original_envs} → {suggested_envs}")
    
    # 2. More aggressive memory cleanup
    def enhanced_cleanup():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        import gc
        gc.collect()
    
    ppo_instance.enhanced_cleanup = enhanced_cleanup
    optimizations.append("Added enhanced memory cleanup")
    
    # 3. Use non-blocking transfers (critical in shared environment)
    def non_blocking_transfer(tensor, device):
        if tensor.device != device:
            return tensor.to(device, non_blocking=True)
        return tensor
    
    ppo_instance.non_blocking_transfer = non_blocking_transfer
    optimizations.append("Added non-blocking device transfers")
    
    # 4. Reduce precision for memory savings
    if torch.cuda.is_available():
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        optimizations.append("Enabled mixed precision optimizations")
    
    return optimizations


def monitor_training_impact(duration_seconds: int = 300):
    """Monitor GPU usage during training to assess impact on other users."""
    print(f"📊 Monitoring GPU usage for {duration_seconds} seconds...")
    
    start_time = time.time()
    measurements = []
    
    while time.time() - start_time < duration_seconds:
        gpus = get_gpu_utilization()
        measurements.append({
            'timestamp': time.time() - start_time,
            'gpus': gpus.copy()
        })
        time.sleep(10)
    
    # Analyze impact
    print("\n📈 GPU Usage Analysis:")
    if measurements:
        for gpu_idx in range(len(measurements[0]['gpus'])):
            utils = [m['gpus'][gpu_idx]['utilization'] for m in measurements]
            memories = [m['gpus'][gpu_idx]['memory_used'] for m in measurements]
            
            avg_util = sum(utils) / len(utils)
            avg_memory = sum(memories) / len(memories) / 1024  # GB
            
            print(f"GPU {gpu_idx}: Avg {avg_util:.1f}% utilization, {avg_memory:.1f}GB memory")


# Example usage for your training
def setup_multi_user_training():
    """Setup training for multi-user environment."""
    print("🎯 MULTI-USER GPU TRAINING SETUP")
    print("=" * 50)
    
    # Step 1: Find best available GPU
    best_gpu = find_best_gpu()
    if best_gpu is not None:
        print(f"📋 Recommended GPU: {best_gpu}")
        print(f"💡 Use: export CUDA_VISIBLE_DEVICES={best_gpu}")
    else:
        print("⚠️ Could not determine best GPU")
    
    # Step 2: Show current status
    gpus = get_gpu_utilization()
    print(f"\n📊 Current GPU Status:")
    for gpu in gpus:
        free_gb = (gpu['memory_total'] - gpu['memory_used']) / 1024
        print(f"  GPU {gpu['index']}: {free_gb:.1f}GB free, {gpu['utilization']}% busy")
    
    # Step 3: Provide recommendations
    print(f"\n🚀 Recommendations for Shared Environment:")
    print("• Use smaller batch sizes to reduce memory pressure")
    print("• Enable non-blocking transfers to reduce queue time") 
    print("• Monitor nvidia-smi during training")
    print("• Consider training during off-peak hours")
    print("• Use gradient accumulation instead of large batches")
    
    return best_gpu


if __name__ == "__main__":
    setup_multi_user_training()


