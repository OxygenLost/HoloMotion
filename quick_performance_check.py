#!/usr/bin/env python3

"""
Quick Performance Analysis for IsaacLab PPO Rollouts

This script provides immediate analysis of potential performance bottlenecks
without requiring a full training run. It examines configuration files,
code patterns, and system setup to identify likely issues.

Usage:
python quick_performance_check.py [config_path]
"""

import sys
import os
import torch
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
from loguru import logger
import psutil
import subprocess


class QuickPerformanceAnalyzer:
    """Analyze performance issues without full training run."""
    
    def __init__(self):
        self.issues_found = []
        self.recommendations = []
        self.system_info = {}
        
    def analyze_system_setup(self):
        """Analyze system configuration for performance issues."""
        logger.info("🖥️  Analyzing system setup...")
        
        issues = []
        
        # GPU Analysis
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            gpu_name = torch.cuda.get_device_name(current_device)
            gpu_memory = torch.cuda.get_device_properties(current_device).total_memory / 1024**3
            
            self.system_info.update({
                'gpu_count': gpu_count,
                'gpu_name': gpu_name,
                'gpu_memory_gb': gpu_memory,
                'cuda_version': torch.version.cuda,
            })
            
            logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            # Check GPU memory
            if gpu_memory < 8:
                issues.append({
                    'category': 'System',
                    'severity': 'HIGH', 
                    'issue': f'Limited GPU memory ({gpu_memory:.1f}GB)',
                    'impact': 'May cause OOM errors or force small batch sizes',
                    'solution': 'Use gradient accumulation, reduce batch size, or upgrade GPU'
                })
            
            # Check CUDA version compatibility
            if torch.version.cuda and torch.version.cuda < '11.0':
                issues.append({
                    'category': 'System',
                    'severity': 'MEDIUM',
                    'issue': f'Old CUDA version ({torch.version.cuda})',
                    'impact': 'May not support latest PyTorch optimizations',
                    'solution': 'Update CUDA to 11.x or 12.x for better performance'
                })
        else:
            issues.append({
                'category': 'System',
                'severity': 'CRITICAL',
                'issue': 'No CUDA GPU available',
                'impact': 'Training will be extremely slow on CPU',
                'solution': 'Install CUDA and use GPU for training'
            })
        
        # CPU Analysis
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / 1024**3
        
        self.system_info.update({
            'cpu_physical_cores': cpu_count,
            'cpu_logical_cores': cpu_count_logical,
            'system_memory_gb': memory_gb,
        })
        
        logger.info(f"CPU: {cpu_count} cores ({cpu_count_logical} threads), {memory_gb:.1f}GB RAM")
        
        if memory_gb < 16:
            issues.append({
                'category': 'System',
                'severity': 'MEDIUM',
                'issue': f'Limited system memory ({memory_gb:.1f}GB)',
                'impact': 'May cause swapping and slow performance',
                'solution': 'Close other applications, use smaller environments count'
            })
        
        # PyTorch settings
        pytorch_info = {
            'version': torch.__version__,
            'num_threads': torch.get_num_threads(),
            'mkl_enabled': torch.backends.mkl.is_available(),
            'cudnn_enabled': torch.backends.cudnn.enabled if torch.cuda.is_available() else False,
        }
        
        self.system_info.update(pytorch_info)
        
        # Check thread settings
        if pytorch_info['num_threads'] > cpu_count_logical:
            issues.append({
                'category': 'System',
                'severity': 'LOW',
                'issue': f'Too many PyTorch threads ({pytorch_info["num_threads"]} > {cpu_count_logical})',
                'impact': 'May cause CPU oversubscription and context switching overhead',
                'solution': f'Set torch.set_num_threads({cpu_count_logical}) or OMP_NUM_THREADS={cpu_count_logical}'
            })
        
        self.issues_found.extend(issues)
        return issues
    
    def analyze_config_file(self, config_path: Optional[str] = None):
        """Analyze configuration files for performance issues."""
        logger.info("📋 Analyzing configuration...")
        
        issues = []
        
        if not config_path:
            # Try to find config files in the project
            config_paths = []
            project_root = Path(__file__).parent
            for config_file in project_root.rglob("*.yaml"):
                if any(keyword in str(config_file).lower() for keyword in ['env', 'algo', 'train']):
                    config_paths.append(config_file)
            
            if not config_paths:
                issues.append({
                    'category': 'Config',
                    'severity': 'HIGH',
                    'issue': 'No configuration files found',
                    'impact': 'Cannot analyze configuration-specific performance issues',
                    'solution': 'Specify config file path to analyze'
                })
                self.issues_found.extend(issues)
                return issues
            
            config_path = config_paths[0]  # Use first found config
        
        try:
            config_path = Path(config_path)
            if not config_path.exists():
                issues.append({
                    'category': 'Config',
                    'severity': 'HIGH',
                    'issue': f'Config file not found: {config_path}',
                    'impact': 'Cannot analyze configuration',
                    'solution': 'Provide correct config file path'
                })
                self.issues_found.extend(issues)
                return issues
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Analyzing config: {config_path}")
            
            # Analyze environment settings
            if 'env' in config:
                env_config = config['env'].get('config', {})
                
                # Check number of environments
                num_envs = env_config.get('num_envs', 1)
                if num_envs > 1000:
                    issues.append({
                        'category': 'Config',
                        'severity': 'MEDIUM',
                        'issue': f'Very high number of environments ({num_envs})',
                        'impact': 'May cause GPU memory issues or slow simulation',
                        'solution': 'Consider reducing num_envs if experiencing performance issues'
                    })
                elif num_envs < 64:
                    issues.append({
                        'category': 'Config',
                        'severity': 'LOW',
                        'issue': f'Low number of environments ({num_envs})',
                        'impact': 'May not fully utilize GPU resources',
                        'solution': 'Consider increasing num_envs for better GPU utilization'
                    })
            
            # Analyze algorithm settings
            if 'algo' in config:
                algo_config = config['algo'].get('config', {})
                
                # Check rollout settings
                num_steps_per_env = algo_config.get('num_steps_per_env', 48)
                if num_steps_per_env > 100:
                    issues.append({
                        'category': 'Config',
                        'severity': 'MEDIUM',
                        'issue': f'High steps per environment ({num_steps_per_env})',
                        'impact': 'Long rollout times, may cause memory issues',
                        'solution': 'Consider reducing num_steps_per_env for faster iterations'
                    })
                
                # Check batch size settings
                num_mini_batches = algo_config.get('num_mini_batches', 6)
                batch_size = (num_envs * num_steps_per_env) // num_mini_batches if 'num_envs' in locals() else None
                
                if batch_size and batch_size < 256:
                    issues.append({
                        'category': 'Config',
                        'severity': 'LOW',
                        'issue': f'Small batch size ({batch_size})',
                        'impact': 'May not efficiently utilize GPU',
                        'solution': 'Consider increasing num_envs or reducing num_mini_batches'
                    })
                
                # Check compilation settings
                compile_model = algo_config.get('compile_model', False)
                if not compile_model:
                    issues.append({
                        'category': 'Config',
                        'severity': 'MEDIUM',
                        'issue': 'Model compilation disabled',
                        'impact': 'Missing potential 20-30% speedup from torch.compile',
                        'solution': 'Set compile_model: true in algorithm config'
                    })
        
        except Exception as e:
            issues.append({
                'category': 'Config',
                'severity': 'HIGH',
                'issue': f'Error reading config file: {e}',
                'impact': 'Cannot analyze configuration',
                'solution': 'Check config file format and syntax'
            })
        
        self.issues_found.extend(issues)
        return issues
    
    def analyze_code_patterns(self):
        """Analyze code for common performance anti-patterns."""
        logger.info("🔍 Analyzing code patterns...")
        
        issues = []
        
        # Check PPO implementation file
        ppo_file = Path(__file__).parent / "holomotion" / "src" / "algo" / "ppo_isaaclab.py"
        
        if ppo_file.exists():
            with open(ppo_file, 'r') as f:
                code_content = f.read()
            
            # Check for device transfer patterns
            if '.to(self.device)' in code_content:
                # Count occurrences in loops
                lines = code_content.split('\n')
                in_loop = False
                device_transfers_in_loop = 0
                
                for line in lines:
                    if any(keyword in line for keyword in ['for ', 'while ']):
                        in_loop = True
                    elif line.strip() == '' or line.startswith('    '):
                        pass  # Continue in loop
                    else:
                        in_loop = False
                    
                    if in_loop and '.to(self.device)' in line:
                        device_transfers_in_loop += 1
                
                if device_transfers_in_loop > 3:
                    issues.append({
                        'category': 'Code',
                        'severity': 'HIGH',
                        'issue': f'Multiple device transfers in loops ({device_transfers_in_loop} found)',
                        'impact': 'Major performance bottleneck from CPU-GPU transfers',
                        'solution': 'Batch device transfers, keep tensors on GPU, use non_blocking=True'
                    })
            
            # Check for synchronous operations
            if 'torch.cuda.synchronize()' in code_content:
                sync_count = code_content.count('torch.cuda.synchronize()')
                if sync_count > 2:
                    issues.append({
                        'category': 'Code',
                        'severity': 'MEDIUM',
                        'issue': f'Frequent CUDA synchronization ({sync_count} calls)',
                        'impact': 'Reduces GPU utilization and parallel execution',
                        'solution': 'Remove unnecessary synchronize() calls, use asynchronous operations'
                    })
            
            # Check for inefficient tensor operations
            inefficient_patterns = [
                ('.cpu()', 'Frequent CPU transfers'),
                ('.numpy()', 'Frequent tensor to numpy conversions'),
                ('for i in range', 'Potential loop that could be vectorized'),
                ('append(', 'List operations that could be batched'),
            ]
            
            for pattern, description in inefficient_patterns:
                count = code_content.count(pattern)
                if count > 5:
                    issues.append({
                        'category': 'Code', 
                        'severity': 'LOW',
                        'issue': f'{description} ({count} occurrences)',
                        'impact': 'May cause performance degradation',
                        'solution': 'Consider vectorizing operations or reducing conversions'
                    })
        
        else:
            issues.append({
                'category': 'Code',
                'severity': 'MEDIUM',
                'issue': 'PPO implementation file not found for analysis',
                'impact': 'Cannot analyze code-specific performance issues',
                'solution': 'Ensure PPO file is in expected location'
            })
        
        self.issues_found.extend(issues)
        return issues
    
    def test_basic_performance(self):
        """Run basic performance tests."""
        logger.info("⚡ Running basic performance tests...")
        
        issues = []
        
        if not torch.cuda.is_available():
            return issues  # Skip GPU tests
        
        # Test GPU-CPU transfer speed
        try:
            device = torch.device('cuda')
            cpu_device = torch.device('cpu')
            
            # Test tensor transfer speed
            test_tensor = torch.randn(1000, 1000, device=cpu_device)
            
            start_time = time.perf_counter()
            gpu_tensor = test_tensor.to(device)
            torch.cuda.synchronize()
            transfer_time = time.perf_counter() - start_time
            
            # Test computation speed
            start_time = time.perf_counter()
            result = torch.matmul(gpu_tensor, gpu_tensor.T)
            torch.cuda.synchronize()
            compute_time = time.perf_counter() - start_time
            
            # Benchmarks (these are rough guidelines)
            if transfer_time > 0.01:  # > 10ms for 4MB transfer
                issues.append({
                    'category': 'Performance',
                    'severity': 'MEDIUM',
                    'issue': f'Slow GPU transfer speed ({transfer_time*1000:.1f}ms for 4MB)',
                    'impact': 'Device transfers will be a major bottleneck',
                    'solution': 'Check GPU connection (PCIe), use pinned memory, batch transfers'
                })
            
            if compute_time > 0.001:  # > 1ms for simple matrix multiply
                issues.append({
                    'category': 'Performance',
                    'severity': 'LOW',
                    'issue': f'Slow GPU computation ({compute_time*1000:.1f}ms for simple matmul)',
                    'impact': 'Neural network inference may be slow',
                    'solution': 'Check GPU utilization, thermal throttling, or driver issues'
                })
            
            # Test memory allocation speed
            start_time = time.perf_counter()
            for _ in range(100):
                temp_tensor = torch.zeros(100, 100, device=device)
                del temp_tensor
            torch.cuda.synchronize()
            alloc_time = time.perf_counter() - start_time
            
            if alloc_time > 0.1:  # > 100ms for 100 allocations
                issues.append({
                    'category': 'Performance',
                    'severity': 'LOW',
                    'issue': f'Slow memory allocation ({alloc_time*1000:.1f}ms for 100 allocs)',
                    'impact': 'Dynamic tensor creation will be slow',
                    'solution': 'Pre-allocate tensors, use memory pools, avoid frequent allocation'
                })
            
        except Exception as e:
            issues.append({
                'category': 'Performance',
                'severity': 'HIGH',
                'issue': f'GPU performance test failed: {e}',
                'impact': 'GPU may not be functioning properly',
                'solution': 'Check GPU installation, drivers, and CUDA setup'
            })
        
        self.issues_found.extend(issues)
        return issues
    
    def generate_report(self):
        """Generate comprehensive performance analysis report."""
        logger.info("📊 Generating performance analysis report...")
        
        # Group issues by severity
        critical_issues = [issue for issue in self.issues_found if issue['severity'] == 'CRITICAL']
        high_issues = [issue for issue in self.issues_found if issue['severity'] == 'HIGH']
        medium_issues = [issue for issue in self.issues_found if issue['severity'] == 'MEDIUM']
        low_issues = [issue for issue in self.issues_found if issue['severity'] == 'LOW']
        
        report = []
        report.append("=" * 80)
        report.append("🚀 ISAAC LAB PPO ROLLOUT PERFORMANCE ANALYSIS")
        report.append("=" * 80)
        
        # System summary
        report.append(f"\n💻 SYSTEM SUMMARY:")
        report.append(f"   GPU: {self.system_info.get('gpu_name', 'Not available')} "
                     f"({self.system_info.get('gpu_memory_gb', 0):.1f}GB)")
        report.append(f"   CPU: {self.system_info.get('cpu_physical_cores', '?')} cores "
                     f"({self.system_info.get('cpu_logical_cores', '?')} threads)")
        report.append(f"   RAM: {self.system_info.get('system_memory_gb', 0):.1f}GB")
        report.append(f"   PyTorch: {self.system_info.get('version', '?')} "
                     f"(CUDA {self.system_info.get('cuda_version', '?')})")
        
        # Issue summary
        total_issues = len(self.issues_found)
        report.append(f"\n🔍 ISSUES FOUND: {total_issues}")
        if critical_issues:
            report.append(f"   🔴 CRITICAL: {len(critical_issues)}")
        if high_issues:
            report.append(f"   🟠 HIGH: {len(high_issues)}")
        if medium_issues:
            report.append(f"   🟡 MEDIUM: {len(medium_issues)}")
        if low_issues:
            report.append(f"   🟢 LOW: {len(low_issues)}")
        
        # Detailed issues
        for severity, issues, emoji in [
            ('CRITICAL', critical_issues, '🔴'),
            ('HIGH', high_issues, '🟠'), 
            ('MEDIUM', medium_issues, '🟡'),
            ('LOW', low_issues, '🟢')
        ]:
            if issues:
                report.append(f"\n{emoji} {severity} PRIORITY ISSUES:")
                report.append("-" * 60)
                
                for i, issue in enumerate(issues, 1):
                    report.append(f"{i}. [{issue['category']}] {issue['issue']}")
                    report.append(f"   Impact: {issue['impact']}")
                    report.append(f"   Solution: {issue['solution']}")
                    report.append("")
        
        # Top recommendations
        report.append("\n🚀 TOP RECOMMENDATIONS FOR IMMEDIATE IMPROVEMENT:")
        report.append("-" * 60)
        
        top_recommendations = []
        if critical_issues or high_issues:
            priority_issues = critical_issues + high_issues
            for issue in priority_issues[:3]:
                top_recommendations.append(f"• {issue['solution']}")
        else:
            # General recommendations if no critical issues
            top_recommendations = [
                "• Enable torch.compile for 20-30% speedup",
                "• Use larger batch sizes to improve GPU utilization", 
                "• Minimize device transfers in rollout loops",
                "• Optimize IsaacLab simulation settings (increase decimation)",
                "• Pre-allocate tensors to avoid memory allocation overhead"
            ]
        
        for rec in top_recommendations:
            report.append(rec)
        
        report.append("\n" + "=" * 80)
        report.append("💡 Run 'python rollout_optimizer.py' for automated optimizations")
        report.append("📊 Use the RolloutProfiler for detailed runtime analysis")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def run_full_analysis(self, config_path: Optional[str] = None):
        """Run complete performance analysis."""
        logger.info("🎯 Starting comprehensive performance analysis...")
        
        # Run all analysis components
        self.analyze_system_setup()
        self.analyze_config_file(config_path)
        self.analyze_code_patterns()
        self.test_basic_performance()
        
        # Generate and print report
        report = self.generate_report()
        print(report)
        
        return {
            'issues_found': self.issues_found,
            'system_info': self.system_info,
            'report': report
        }


def main():
    """Main entry point for quick performance analysis."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    
    analyzer = QuickPerformanceAnalyzer()
    results = analyzer.run_full_analysis(config_path)
    
    # Save report to file
    report_file = Path("performance_analysis_report.txt")
    with open(report_file, 'w') as f:
        f.write(results['report'])
    
    logger.info(f"📄 Full report saved to: {report_file}")
    
    return results


if __name__ == "__main__":
    main()


