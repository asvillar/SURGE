import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import joblib


def posix_str(path) -> str:
    """Return ``path`` as a forward-slash string for portable persistence.

    Artifact/summary JSON (and the canvas data derived from it) is often
    generated on one OS and read on another. Forward slashes are accepted by
    Windows, macOS and Linux alike and avoid the backslash-escaping problems
    that native Windows paths cause inside JSON, so we normalise every path we
    *write* to disk with this helper.
    """
    return Path(path).as_posix()


# Guard torch import for environments without PyTorch
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# Try to import psutil with fallback for compute resource analysis
try:
    import psutil
except ImportError:
    psutil = None

# Try to import matplotlib for plotting
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    # Still need numpy for resource summary calculations
    try:
        import numpy as np
    except ImportError:
        raise ImportError("numpy is required for SURGE. Install with: pip install numpy")


class ResourceMonitor:
    """Monitor system resources during training and hyperparameter tuning"""
    
    def __init__(self):
        if psutil is None:
            raise ImportError("psutil is required for resource monitoring. Install with: pip install psutil")
        self.reset()
    
    def reset(self):
        """Reset monitoring data"""
        self.timestamps = []
        self.ram_usage = []
        self.cpu_usage = []
        self.peak_ram = 0
        self.start_time = time.time()
        
    def update(self):
        """Update resource metrics and return current values"""
        # Get current metrics
        assert psutil is not None, "psutil is required for resource monitoring"
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        ram_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        cpu_percent = process.cpu_percent()
        
        # Update peak RAM
        self.peak_ram = max(self.peak_ram, ram_mb)
        
        # Store metrics
        self.timestamps.append(time.time() - self.start_time)
        self.ram_usage.append(ram_mb)
        self.cpu_usage.append(cpu_percent)
        
        return ram_mb, cpu_percent
    
    def get_summary(self):
        """Get resource usage summary statistics"""
        if not self.ram_usage:
            return "No data collected"
        
        return {
            'peak_ram_mb': self.peak_ram,
            'avg_ram_mb': np.mean(self.ram_usage) if self.ram_usage else 0,
            'avg_cpu_percent': np.mean(self.cpu_usage) if self.cpu_usage else 0,
            'duration_sec': self.timestamps[-1] if self.timestamps else 0
        }
    
    def plot_resources(self, title="System Resource Monitoring"):
        """Plot resource usage over time"""
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ Matplotlib not available for plotting. Install with: pip install matplotlib")
            return
        
        if len(self.timestamps) < 2:
            print("Insufficient data for plotting")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # RAM usage plot
        ax1.plot(self.timestamps, self.ram_usage, 'b-', linewidth=2, label='RAM Usage')
        ax1.axhline(y=self.peak_ram, color='r', linestyle='--', alpha=0.7, 
                   label=f'Peak: {self.peak_ram:.1f} MB')
        ax1.set_ylabel('RAM Usage (MB)')
        ax1.set_title(title)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # CPU usage plot
        ax2.plot(self.timestamps, self.cpu_usage, 'g-', linewidth=2, label='CPU Usage')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('CPU Usage (%)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()


def get_device():
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for get_device(). Install with: pip install torch")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(model, path):
    joblib.dump(model, path)


def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def setup_surge_path():
    """
    Check if SURGE environment variable is set and create it if needed.

    Returns:
        Path: The SURGE repository path

    Raises:
        FileNotFoundError: If SURGE path cannot be determined or doesn't exist
    """
    # Check if SURGE environment variable exists
    surge_path = os.environ.get('SURGE')

    if surge_path is None:
        # Try to determine SURGE path from current file location
        current_file = Path(__file__).resolve()
        # Navigate up from surge/utils.py to find the repo root
        potential_surge_path = current_file.parent.parent

        # Verify this is indeed the SURGE repo by checking for key directories
        if (potential_surge_path / 'surge').exists() and (potential_surge_path / 'notebooks').exists():
            surge_path = str(potential_surge_path)
            # Set the environment variable for this session and future use
            os.environ['SURGE'] = surge_path
            print(f"✅ SURGE environment variable set to: {surge_path}")
        else:
            raise FileNotFoundError(
                "Could not automatically determine SURGE repository path. "
                "Please set the SURGE environment variable manually: "
                "export SURGE=/path/to/your/SURGE/repo"
            )
    else:
        print(f"✅ Using existing SURGE environment variable: {surge_path}")

    # Validate the path exists
    surge_path_obj = Path(surge_path)
    if not surge_path_obj.exists():
        raise FileNotFoundError(f"SURGE path does not exist: {surge_path}")

    return surge_path_obj


def get_data_path(dataset_name="HHFW-NSTX"):
    """
    Get the path to a specific dataset in the SURGE repository.

    Args:
        dataset_name (str): Name of the dataset directory

    Returns:
        Path: Path to the dataset directory
    """
    from surge.benchmarks import paths

    datasets_dir = paths.datasets_root()
    data_path = datasets_dir / dataset_name

    if not data_path.exists():
        available_datasets = []
        if datasets_dir.exists():
            available_datasets = [d.name for d in datasets_dir.iterdir() if d.is_dir()]

        raise FileNotFoundError(
            f"Dataset '{dataset_name}' not found at {data_path}.\n"
            f"Available datasets: {available_datasets}"
        )

    return data_path


def detect_compute_resources():
    """Detect available compute resources and current usage"""
    
    print("🔍 SYSTEM RESOURCE DETECTION")
    print("=" * 60)
    
    # System info
    print(f"💻 System: {platform.system()} {platform.release()}")
    print(f"🏗️ Architecture: {platform.machine()}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    
    if psutil is None:
        print("\n⚠️ Warning: psutil not available. Limited resource detection.")
        return {
            'cpu_cores_physical': None,
            'cpu_cores_logical': None,
            'total_ram_gb': None,
            'gpu_available': False,
            'gpu_info': 'psutil not available',
            'device': 'cpu'
        }
    
    # CPU Information
    cpu_count_physical = psutil.cpu_count(logical=False)
    cpu_count_logical = psutil.cpu_count(logical=True)
    
    # Try to get CPU frequency (may not work on all systems)
    try:
        cpu_freq = psutil.cpu_freq()
    except (FileNotFoundError, AttributeError):
        cpu_freq = None
    
    print(f"\n🧠 CPU Information:")
    print(f"   Physical Cores: {cpu_count_physical}")
    print(f"   Logical Cores: {cpu_count_logical}")
    if cpu_freq:
        print(f"   Frequency: {cpu_freq.current:.2f} MHz (max: {cpu_freq.max:.2f} MHz)")
    else:
        print(f"   Frequency: Not available (common on Apple Silicon/ARM)")
    
    # Memory Information
    memory = psutil.virtual_memory()
    print(f"\n💾 Memory Information:")
    print(f"   Total RAM: {memory.total / (1024**3):.2f} GB")
    print(f"   Available RAM: {memory.available / (1024**3):.2f} GB")
    print(f"   Used RAM: {memory.used / (1024**3):.2f} GB ({memory.percent:.1f}%)")
    
    # GPU Detection
    gpu_available = False
    gpu_info = "None detected"
    
    # Try to detect NVIDIA GPU
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,utilization.gpu', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            gpu_lines = result.stdout.strip().split('\n')
            gpu_available = True
            gpu_info = f"NVIDIA GPU(s) detected: {len(gpu_lines)} device(s)"
            for i, line in enumerate(gpu_lines):
                parts = line.split(', ')
                if len(parts) >= 3:
                    name, memory_mb, util = parts[0], parts[1], parts[2]
                    print(f"   GPU {i}: {name} ({memory_mb} MB, {util}% util)")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Try to detect Metal (macOS) or other GPUs
    if not gpu_available:
        try:
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], 
                                      capture_output=True, text=True, timeout=5)
                if "Metal" in result.stdout or "GPU" in result.stdout:
                    gpu_available = True
                    gpu_info = "Metal GPU detected (macOS)"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    
    print(f"\n🎮 GPU Information:")
    print(f"   Status: {gpu_info}")
    
    # Check if PyTorch can use GPU
    if TORCH_AVAILABLE:
        cuda_available = torch.cuda.is_available()
        mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        print(f"\n🔥 PyTorch GPU Support:")
        print(f"   CUDA Available: {cuda_available}")
        if cuda_available:
            print(f"   CUDA Devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"     Device {i}: {torch.cuda.get_device_name(i)}")
        print(f"   MPS (Metal) Available: {mps_available}")
        
        if cuda_available:
            device = "cuda"
        elif mps_available:
            device = "mps"
        else:
            device = "cpu"
        print(f"   Recommended Device: {device}")
    else:
        print(f"\n🔥 PyTorch: Not installed")
        device = "cpu"
    
    # Detect common cluster/job scheduler environment variables to support
    # multi-node workstation and cluster detection (SLURM, PBS, LSF, Cobalt)
    job_info = None
    try:
        if 'SLURM_JOB_NODELIST' in os.environ:
            job_info = {
                'scheduler': 'slurm',
                'nodelist': os.environ.get('SLURM_JOB_NODELIST'),
                'nnodes': int(os.environ.get('SLURM_NNODES', '1')),
            }
        elif 'PBS_NODEFILE' in os.environ:
            nodefile = os.environ.get('PBS_NODEFILE')
            job_info = {'scheduler': 'pbs', 'nodefile': nodefile}
            # Attempt to count nodes if file exists
            try:
                if nodefile and os.path.exists(nodefile):
                    with open(nodefile, 'r') as fh:
                        job_info['nnodes'] = sum(1 for _ in fh)
                else:
                    job_info['nnodes'] = None
            except Exception:
                job_info['nnodes'] = None
        elif 'LSB_JOBID' in os.environ or 'LSB_HOSTS' in os.environ:
            job_info = {'scheduler': 'lsf'}
        elif 'COBALT_JOBID' in os.environ:
            job_info = {'scheduler': 'cobalt'}
    except Exception:
        job_info = None

    return {
        'cpu_cores_physical': cpu_count_physical,
        'cpu_cores_logical': cpu_count_logical,
        'total_ram_gb': memory.total / (1024**3),
        'gpu_available': gpu_available,
        'gpu_info': gpu_info,
        'device': device if 'device' in locals() else 'cpu',
        'job_info': job_info,
    }


def enhanced_cpu_monitoring():
    """Enhanced CPU monitoring to capture actual usage during training"""
    
    print("\n🔍 REAL-TIME CPU MONITORING TEST")
    print("=" * 60)
    
    if psutil is None:
        print("⚠️ Warning: psutil not available. Cannot monitor CPU usage.")
        return {'ok': False, 'reason': 'psutil unavailable'}
    
    # Baseline CPU usage
    baseline_cpu = psutil.cpu_percent(interval=1)
    # Per-core CPU usage
    per_core_cpu = psutil.cpu_percent(interval=1, percpu=True)

    # Test with a brief computation
    print("\nTesting CPU monitoring during computation...")
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor

    # Create dummy data for CPU test
    X_test = np.random.random((500, 8))
    y_test = np.random.random(500)

    # Monitor CPU during training
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=40, n_jobs=-1, random_state=42)

    pre_fit_cpu = psutil.cpu_percent(interval=None)
    model.fit(X_test, y_test)
    post_fit_cpu = psutil.cpu_percent(interval=0.5)

    training_time = time.time() - start_time

    cpu_metrics = {
        'ok': True,
        'baseline_cpu_percent': float(baseline_cpu),
        'per_core_cpu_percent': [float(x) for x in per_core_cpu],
        'pre_fit_cpu_percent': float(pre_fit_cpu),
        'post_fit_cpu_percent': float(post_fit_cpu),
        'training_time_sec': float(training_time),
        'utilization_detected': bool(post_fit_cpu > pre_fit_cpu + 5)
    }

    print(f"Baseline CPU usage: {baseline_cpu:.1f}%")
    print(f"Per-core sample: {[f'{cpu:.1f}%' for cpu in per_core_cpu]}")
    print(f"CPU usage during test: {pre_fit_cpu:.1f}% → {post_fit_cpu:.1f}%")
    print(f"Training time: {training_time:.3f} seconds")

    return cpu_metrics


def run_compute_resource_analysis(verbose=1):
    """Run complete enhanced compute resource analysis.

    Args:
        verbose (int): 0 for compact summary, 1 for detailed tables (default).

    Returns:
        tuple: (system_info, cpu_metrics)
    """

    system_info = detect_compute_resources()
    cpu_metrics = enhanced_cpu_monitoring()

    # Build a richer summary dict
    summary = {
        'system_info': system_info,
        'cpu_metrics': cpu_metrics,
        'timestamp': time.time()
    }

    if verbose <= 0:
        # Compact single-line/table summary: cores, RAM, GPU, device, baseline/post-fit
        try:
            mod = system_info
            baseline = cpu_metrics.get('baseline_cpu_percent') if isinstance(cpu_metrics, dict) else None
            post = cpu_metrics.get('post_fit_cpu_percent') if isinstance(cpu_metrics, dict) else None
            gpu = mod.get('gpu_available')
            device = mod.get('device')
            cores = mod.get('cpu_cores_logical')
            ram = mod.get('total_ram_gb')
            # Print compact table with labeled sections
            print('\nCOMPUTE SUMMARY (compact)')
            print('----------------------------------------')
            # System & architecture
            print('System & Architecture:')
            print(f"  OS: {platform.system()} {platform.release()} | Arch: {platform.machine()} | Python: {sys.version.split()[0]}")
            print('')
            # CPU
            phys = mod.get('cpu_cores_physical')
            print('CPU:')
            print(f"  Logical cores: {cores} | Physical cores: {phys}")
            if baseline is not None:
                print(f"  Baseline CPU %: {baseline:.1f}% | Post-fit CPU %: {post:.1f}%")
            print('')
            # GPU
            print('GPU:')
            print(f"  Available: {gpu} | Info: {mod.get('gpu_info')} | Recommended device: {device}")
            print('')
            # Memory
            print('Memory (RAM):')
            try:
                print(f"  Total: {ram:.2f} GB | Available: {mod.get('total_ram_gb'):.2f} GB")
            except Exception:
                print(f"  Total: {ram} GB")
            # If job_info present, print short cluster info
            ji = mod.get('job_info')
            if ji:
                print('')
                print('Cluster/Job info:')
                print(f"  Scheduler: {ji.get('scheduler')} | nodes: {ji.get('nnodes')}")
        except Exception:
            print('Failed to print compact compute summary')
        return system_info, cpu_metrics

    # verbose >=1 => detailed printing
    print("🚀 ENHANCED COMPUTE RESOURCE ANALYSIS")
    print("=" * 80)
    try:
        pretty_print_resource_summary(system_info, cpu_metrics)
    except Exception:
        print('\nSummary:')
        print(system_info)
        print(cpu_metrics)

    # Model recommendations (experimental)
    print('\nMODEL RECOMMENDATIONS (experimental - under construction)')
    print('-' * 60)
    try:
        if system_info.get('gpu_available') and system_info.get('device') and system_info.get('device') != 'cpu':
            print(' - GPU present: prefer PyTorch models to leverage GPU acceleration (MPS/CUDA).')
            print('   Example: use PyTorch MLP/NN models and move tensors to device.')
        else:
            print(' - No suitable GPU detected: scikit-learn RandomForest or CPU-based PyTorch is recommended.')
            print('   Example: RandomForest with n_jobs=-1 for parallel training.')
        print(' - Consider model size vs available RAM; if RAM < 8GB use smaller models or batch training.')
        print(' - Note: these model recommendations are experimental and under construction; validate before production.')
    except Exception:
        print(' - Model recommendations: unavailable')

    return system_info, cpu_metrics


def pretty_print_resource_summary(system_info, cpu_metrics):
    """Print a pair of aligned ASCII tables summarizing system_info and cpu_metrics."""
    # System table
    rows = [
        ('Physical cores', str(system_info.get('cpu_cores_physical'))),
        ('Logical cores', str(system_info.get('cpu_cores_logical'))),
        ('Total RAM (GB)', f"{system_info.get('total_ram_gb', 0):.2f}"),
        ('GPU available', str(system_info.get('gpu_available'))),
        ('GPU info', str(system_info.get('gpu_info'))),
        ('Recommended device', str(system_info.get('device'))),
    ]

    col_left = max(len(r[0]) for r in rows)
    col_right = max(len(r[1]) for r in rows)

    print('\nSYSTEM')
    print('-' * (col_left + col_right + 5))
    for k, v in rows:
        print(f"{k.ljust(col_left)} : {v.rjust(col_right)}")
    print('-' * (col_left + col_right + 5))

    # CPU metrics table
    print('\nCPU MONITORING')
    if not cpu_metrics or not cpu_metrics.get('ok'):
        print('  CPU monitoring not available or failed: ', cpu_metrics.get('reason') if isinstance(cpu_metrics, dict) else cpu_metrics)
        return

    per_core = cpu_metrics.get('per_core_cpu_percent', [])
    per_core_sample = ', '.join([f"{x:.0f}%" for x in per_core[:8]])

    cpu_rows = [
        ('Baseline CPU %', f"{cpu_metrics.get('baseline_cpu_percent'):.1f}%"),
        ('Pre-fit CPU %', f"{cpu_metrics.get('pre_fit_cpu_percent'):.1f}%"),
        ('Post-fit CPU %', f"{cpu_metrics.get('post_fit_cpu_percent'):.1f}%"),
        ('Training time (s)', f"{cpu_metrics.get('training_time_sec'):.2f}"),
        ('Utilization detected', str(cpu_metrics.get('utilization_detected'))),
        ('Per-core sample', per_core_sample),
    ]

    col_left = max(len(r[0]) for r in cpu_rows)
    col_right = max(len(r[1]) for r in cpu_rows)
    print('-' * (col_left + col_right + 5))
    for k, v in cpu_rows:
        print(f"{k.ljust(col_left)} : {v.rjust(col_right)}")
    print('-' * (col_left + col_right + 5))

    # Recommendations
    print('\nRECOMMENDATIONS')
    print('-' * 40)


def _collect_system_info_quiet():
    """Collect system info without printing (used by tabular reporters)."""
    info = {
        'os': f"{platform.system()} {platform.release()}",
        'architecture': platform.machine(),
        'python': sys.version.split()[0],
        'cpu_cores_physical': None,
        'cpu_cores_logical': None,
        'total_ram_gb': None,
        'gpu_available': False,
        'gpu_info': 'None detected',
        'device': 'cpu',
        'job_info': None,
    }

    # If psutil available, collect live usage numbers
    if psutil is None:
        return info

    try:
        info['cpu_cores_physical'] = psutil.cpu_count(logical=False)
        info['cpu_cores_logical'] = psutil.cpu_count(logical=True)
    except Exception:
        pass

    # CPU brand/type
    try:
        cpu_brand = platform.processor() or ''
        if not cpu_brand and platform.system() == 'Darwin':
            # macOS best-effort
            try:
                cpu_brand = subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
            except Exception:
                cpu_brand = ''
        info['cpu_brand'] = cpu_brand or 'Unknown'
    except Exception:
        info['cpu_brand'] = 'Unknown'

    try:
        info['cpu_percent'] = psutil.cpu_percent(interval=1)
        info['per_core_percent'] = psutil.cpu_percent(interval=1, percpu=True)
    except Exception:
        info['cpu_percent'] = None
        info['per_core_percent'] = []

    try:
        memory = psutil.virtual_memory()
        info['total_ram_gb'] = memory.total / (1024**3)
        info['memory_available_gb'] = memory.available / (1024**3)
        info['memory_used_gb'] = memory.used / (1024**3)
        info['memory_percent'] = memory.percent
    except Exception:
        info['memory_available_gb'] = None
        info['memory_used_gb'] = None
        info['memory_percent'] = None

    # GPU detection (best-effort, quiet) - produce gpu_details list with dict per device
    gpu_details = []
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True, timeout=3)
        if result.returncode == 0 and result.stdout.strip():
            lines = [l for l in result.stdout.strip().split('\n') if l]
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    name = parts[0]
                    mem_total = float(parts[1])
                    mem_used = float(parts[2])
                    util = float(parts[3])
                    gpu_details.append({'name': name, 'memory_total_mb': mem_total, 'memory_used_mb': mem_used, 'util_percent': util})
            if gpu_details:
                info['gpu_available'] = True
                info['gpu_info'] = f"NVIDIA: {len(gpu_details)} device(s)"
    except Exception:
        pass

    # macOS Metal detection
    try:
        if not gpu_details and platform.system() == 'Darwin':
            result = subprocess.run(['system_profiler', 'SPDisplaysDataType'], capture_output=True, text=True, timeout=3)
            if result.stdout and ('Metal' in result.stdout or 'GPU' in result.stdout):
                info['gpu_available'] = True
                info['gpu_info'] = 'Metal GPU detected (macOS)'
                gpu_details.append({'name': 'MPS/Metal', 'memory_total_mb': None, 'memory_used_mb': None, 'util_percent': None})
    except Exception:
        pass

    # PyTorch device detection (quiet)
    if TORCH_AVAILABLE:
        try:
            cuda_available = torch.cuda.is_available()
            mps_available = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
            if cuda_available:
                info['gpu_available'] = True
                info['device'] = 'cuda'
                info['gpu_info'] = 'CUDA available'
                # add GPU names if not already
                try:
                    for i in range(torch.cuda.device_count()):
                        gpu_details.append({'name': torch.cuda.get_device_name(i), 'memory_total_mb': None, 'memory_used_mb': None, 'util_percent': None})
                except Exception:
                    pass
            elif mps_available:
                info['gpu_available'] = True
                info['device'] = 'mps'
                info['gpu_info'] = 'MPS/Metal available'
                if not any(g.get('name') == 'MPS/Metal' for g in gpu_details):
                    gpu_details.append({'name': 'MPS/Metal', 'memory_total_mb': None, 'memory_used_mb': None, 'util_percent': None})
        except Exception:
            pass

    # (NVML/pynvml probing was removed per user request)
    info['gpu_details'] = gpu_details

    # Job/cluster detection (quiet)
    try:
        if 'SLURM_JOB_NODELIST' in os.environ:
            info['job_info'] = {'scheduler': 'slurm', 'nodelist': os.environ.get('SLURM_JOB_NODELIST'), 'nnodes': int(os.environ.get('SLURM_NNODES', '1'))}
        elif 'PBS_NODEFILE' in os.environ:
            nodefile = os.environ.get('PBS_NODEFILE')
            ji = {'scheduler': 'pbs', 'nodefile': nodefile}
            try:
                if nodefile and os.path.exists(nodefile):
                    with open(nodefile, 'r') as fh:
                        ji['nnodes'] = sum(1 for _ in fh)
                else:
                    ji['nnodes'] = None
            except Exception:
                ji['nnodes'] = None
            info['job_info'] = ji
        elif 'LSB_JOBID' in os.environ or 'LSB_HOSTS' in os.environ:
            info['job_info'] = {'scheduler': 'lsf'}
        elif 'COBALT_JOBID' in os.environ:
            info['job_info'] = {'scheduler': 'cobalt'}
    except Exception:
        pass

    return info


def system_resource_report(save: bool = False, save_dir: Optional[str] = None):
    """Produce a concise tabular system report (suitable before running SURGE).

    Prints only an ASCII table and returns the collected info dict.

    Args:
        save (bool): If True, save a JSON and text copy to `surge_dev_artifacts` (default: False).
        save_dir (str|None): Optional directory path to save artifacts. If None, uses
            $SURGE/surge_dev_artifacts or ./surge_dev_artifacts.
    """
    import io
    info = _collect_system_info_quiet()

    rows = [
        ('OS', info.get('os')),
        ('Architecture', info.get('architecture')),
        ('Python', info.get('python')),
        # CPU summary will be printed in its own section
        ('Physical cores', str(info.get('cpu_cores_physical'))),
        ('Logical cores', str(info.get('cpu_cores_logical'))),
        ('Total RAM (GB)', f"{info.get('total_ram_gb'):.2f}" if info.get('total_ram_gb') is not None else 'None'),
        ('GPU available', str(info.get('gpu_available'))),
        ('GPU info', str(info.get('gpu_info'))),
        ('Recommended device', str(info.get('device'))),
    ]

    col_left = max(len(r[0]) for r in rows)
    col_right = max(len(r[1]) for r in rows)

    sio = io.StringIO()
    header = 'SYSTEM RESOURCE REPORT'
    print(header, file=sio)
    print('-' * (col_left + col_right + 5), file=sio)
    for k, v in rows:
        print(f"{k.ljust(col_left)} : {v.rjust(col_right)}", file=sio)
    print('-' * (col_left + col_right + 5), file=sio)

    # Short job info line if present
    ji = info.get('job_info')
    if ji:
        print(f"Job scheduler: {ji.get('scheduler')} | nodes: {ji.get('nnodes')}", file=sio)

    out = sio.getvalue()
    # Print a single boxed ASCII table with sections in order: CPU -> GPU -> MEMORY
    # Prepare grouped rows
    cpu_rows = []
    cpu_brand = info.get('cpu_brand', 'Unknown')
    cpu_rows.append(('Type', str(cpu_brand)))
    cpu_rows.append(('Physical cores', str(info.get('cpu_cores_physical'))))
    cpu_rows.append(('Logical cores', str(info.get('cpu_cores_logical'))))
    cpu_pct = info.get('cpu_percent')
    cpu_rows.append(('Current usage', f"{cpu_pct:.1f}%" if cpu_pct is not None else 'N/A'))

    gpu_rows = []
    gpu_rows.append(('Available', str(info.get('gpu_available'))))
    gpu_rows.append(('Info', str(info.get('gpu_info'))))
    gd = info.get('gpu_details', [])
    for i, g in enumerate(gd):
        name = g.get('name') or f'GPU {i}'
        parts = []
        if g.get('memory_total_mb') is not None:
            parts.append(f"mem {g.get('memory_total_mb'):.0f}MB")
        if g.get('memory_used_mb') is not None:
            parts.append(f"used {g.get('memory_used_mb'):.0f}MB")
        if g.get('util_percent') is not None:
            parts.append(f"util {g.get('util_percent'):.0f}%")
        extra = ', '.join(parts) if parts else ''
        gpu_rows.append((f'{name}', extra))

    mem_rows = []
    tr = info.get('total_ram_gb')
    av = info.get('memory_available_gb')
    us = info.get('memory_used_gb')
    mp = info.get('memory_percent')
    mem_rows.append(('Total (GB)', f"{tr:.2f}" if tr is not None else 'N/A'))
    mem_rows.append(('Available (GB)', f"{av:.2f}" if av is not None else 'N/A'))
    mem_rows.append(('Used (GB)', f"{us:.2f} ({mp:.1f}%)" if us is not None and mp is not None else 'N/A'))

    # Compute widths
    all_rows = [('CPU ' + k, v) for k, v in cpu_rows] + [('GPU ' + k, v) for k, v in gpu_rows] + [('MEM ' + k, v) for k, v in mem_rows]
    col_left2 = max(len(r[0]) for r in all_rows)
    col_right2 = max(len(r[1]) for r in all_rows)
    total_width = col_left2 + 3 + col_right2

    def hline():
        print('+' + '-' * total_width + '+')

    def print_row(left, right):
        print('| ' + left.ljust(col_left2) + ' : ' + right.rjust(col_right2) + ' |')

    # Print header
    hline()
    print('| ' + header.center(total_width - 2) + ' |')
    hline()

    # CPU section
    print('| ' + 'CPU'.center(total_width - 2) + ' |')
    hline()
    for k, v in cpu_rows:
        print_row(k, v)
    hline()

    # GPU section
    print('| ' + 'GPU'.center(total_width - 2) + ' |')
    hline()
    for k, v in gpu_rows:
        print_row(k, v)
    hline()

    # MEMORY section
    print('| ' + 'MEMORY'.center(total_width - 2) + ' |')
    hline()
    for k, v in mem_rows:
        print_row(k, v)
    hline()

    # Optionally save
    if save:
        try:
            from pathlib import Path
            artifacts = Path(save_dir) if save_dir else Path(os.environ.get('SURGE', '.')) / 'surge_dev_artifacts'
            artifacts.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            json_path = artifacts / f'system_resource_report_{ts}.json'
            txt_path = artifacts / f'system_resource_report_{ts}.txt'
            with open(json_path, 'w') as fh:
                json.dump(info, fh, indent=2)
            with open(txt_path, 'w') as fh:
                fh.write(out)
        except Exception:
            # non-fatal: just continue
            pass

    return info


def system_monitor_report(run_cpu_test: bool = False, n_estimators: int = 40, n_jobs: int = -1, sample_rows: int = 500, save: bool = False, save_dir: Optional[str] = None):
    """Produce a concise tabular runtime monitor report (CPU/GPU) while building models.

    Args:
        run_cpu_test (bool): If True, run a short RandomForest fit to measure CPU utilization.
        n_estimators (int): n_estimators for the brief RandomForest (if run_cpu_test).
        n_jobs (int): n_jobs passed to the RandomForest (if run_cpu_test).
        sample_rows (int): Number of dummy rows for the short CPU test.

    Returns:
        dict: cpu_metrics-like dict with monitoring results.
    """
    import io
    if psutil is None:
        metrics = {'ok': False, 'reason': 'psutil unavailable'}
        sio = io.StringIO()
        print('CPU MONITOR REPORT', file=sio)
        print('------------------', file=sio)
        print('  psutil not available', file=sio)
        out = sio.getvalue()
        print(out)
        if save:
            try:
                from pathlib import Path
                artifacts = Path(save_dir) if save_dir else Path(os.environ.get('SURGE', '.')) / 'surge_dev_artifacts'
                artifacts.mkdir(parents=True, exist_ok=True)
                ts = int(time.time())
                txt_path = artifacts / f'system_monitor_report_{ts}.txt'
                with open(txt_path, 'w') as fh:
                    fh.write(out)
            except Exception:
                pass
        return metrics

    baseline = psutil.cpu_percent(interval=1)
    per_core = psutil.cpu_percent(interval=1, percpu=True)

    metrics = {
        'ok': True,
        'baseline_cpu_percent': float(baseline),
        'per_core_cpu_percent': [float(x) for x in per_core],
    }

    # Optionally run a brief CPU test
    if run_cpu_test:
        try:
            import numpy as _np
            from sklearn.ensemble import RandomForestRegressor as _RFR

            X = _np.random.random((sample_rows, 8))
            y = _np.random.random(sample_rows)
            start = time.time()
            pre = psutil.cpu_percent(interval=None)
            m = _RFR(n_estimators=n_estimators, n_jobs=n_jobs, random_state=42)
            m.fit(X, y)
            post = psutil.cpu_percent(interval=0.5)
            dur = time.time() - start

            metrics.update({
                'pre_fit_cpu_percent': float(pre),
                'post_fit_cpu_percent': float(post),
                'training_time_sec': float(dur),
                'utilization_detected': bool(post > pre + 5),
            })
        except Exception as e:
            metrics.update({'ok': False, 'reason': f'cpu_test_failed: {e}'})

    # Also include GPU/device info quietly
    sysinfo = _collect_system_info_quiet()
    metrics['gpu_available'] = sysinfo.get('gpu_available')
    metrics['device'] = sysinfo.get('device')
    # (NVML/pynvml probing was removed per user request)

    # Prepare tabular output
    # Prepare boxed report (CPU -> GPU -> MEMORY) using sysinfo + metrics
    import io as _io
    sio = _io.StringIO()

    cpu_rows = []
    cpu_rows.append(('Type', str(sysinfo.get('cpu_brand', 'Unknown'))))
    cpu_rows.append(('Physical cores', str(sysinfo.get('cpu_cores_physical'))))
    cpu_rows.append(('Logical cores', str(sysinfo.get('cpu_cores_logical'))))
    cpu_pct = sysinfo.get('cpu_percent') if sysinfo.get('cpu_percent') is not None else metrics.get('baseline_cpu_percent')
    cpu_rows.append(('Current usage', f"{cpu_pct:.1f}%" if cpu_pct is not None else 'N/A'))
    # include baseline/per-core as extra rows
    cpu_rows.append(('Baseline CPU %', f"{metrics.get('baseline_cpu_percent'):.1f}%"))
    per_core_sample = ', '.join([f"{int(x)}%" for x in metrics.get('per_core_cpu_percent', [])[:8]])
    cpu_rows.append(('Per-core sample', per_core_sample))

    gpu_rows = []
    gpu_rows.append(('Available', str(metrics.get('gpu_available'))))
    gpu_rows.append(('Info', str(sysinfo.get('gpu_info'))))
    gd = sysinfo.get('gpu_details', [])
    for i, g in enumerate(gd):
        name = g.get('name') or f'GPU {i}'
        parts = []
        if g.get('memory_total_mb') is not None:
            parts.append(f"mem {g.get('memory_total_mb'):.0f}MB")
        if g.get('memory_used_mb') is not None:
            parts.append(f"used {g.get('memory_used_mb'):.0f}MB")
        if g.get('util_percent') is not None:
            parts.append(f"util {g.get('util_percent'):.0f}%")
        extra = ', '.join(parts) if parts else ''
        gpu_rows.append((f'{name}', extra))

    mem_rows = []
    tr = sysinfo.get('total_ram_gb')
    av = sysinfo.get('memory_available_gb')
    us = sysinfo.get('memory_used_gb')
    mp = sysinfo.get('memory_percent')
    mem_rows.append(('Total (GB)', f"{tr:.2f}" if tr is not None else 'N/A'))
    mem_rows.append(('Available (GB)', f"{av:.2f}" if av is not None else 'N/A'))
    mem_rows.append(('Used (GB)', f"{us:.2f} ({mp:.1f}%)" if us is not None and mp is not None else 'N/A'))

    all_rows = [('CPU ' + k, v) for k, v in cpu_rows] + [('GPU ' + k, v) for k, v in gpu_rows] + [('MEM ' + k, v) for k, v in mem_rows]
    col_left2 = max(len(r[0]) for r in all_rows)
    col_right2 = max(len(r[1]) for r in all_rows)
    total_width = col_left2 + 3 + col_right2

    def hline():
        sio.write('+' + '-' * total_width + '+' + '\n')

    def print_row(left, right):
        sio.write('| ' + left.ljust(col_left2) + ' : ' + right.rjust(col_right2) + ' |' + '\n')

    header = 'SYSTEM MONITOR REPORT'
    hline()
    sio.write('| ' + header.center(total_width - 2) + ' |' + '\n')
    hline()

    sio.write('| ' + 'CPU'.center(total_width - 2) + ' |' + '\n')
    hline()
    for k, v in cpu_rows:
        print_row(k, v)
    hline()

    sio.write('| ' + 'GPU'.center(total_width - 2) + ' |' + '\n')
    hline()
    for k, v in gpu_rows:
        print_row(k, v)
    hline()

    sio.write('| ' + 'MEMORY'.center(total_width - 2) + ' |' + '\n')
    hline()
    for k, v in mem_rows:
        print_row(k, v)
    hline()

    out = sio.getvalue()
    print(out)

    if save:
        try:
            from pathlib import Path
            artifacts = Path(save_dir) if save_dir else Path(os.environ.get('SURGE', '.')) / 'surge_dev_artifacts'
            artifacts.mkdir(parents=True, exist_ok=True)
            ts = int(time.time())
            json_path = artifacts / f'system_monitor_report_{ts}.json'
            txt_path = artifacts / f'system_monitor_report_{ts}.txt'
            with open(json_path, 'w') as fh:
                json.dump(metrics, fh, indent=2)
            with open(txt_path, 'w') as fh:
                fh.write(out)
        except Exception:
            pass

    return metrics


def inspect_surge(module_names=None, verbose=0):
    """
    Inspect SURGE modules for quick health checks.

    Args:
        module_names (list[str] | None): List of module names to check. If None,
            a sensible default set is used.
        verbose (int): Verbosity level. 0 => concise summary per module.
                       1 => print short details while running.

    Returns:
        dict: Mapping module name -> info dict with keys: loaded (bool),
              traceback (str|None), summary (short str), classes, functions,
              variables, top (list of top public names).
    """
    import importlib
    import inspect
    import traceback

    # If module_names provided, we'll inspect those too, but we'll always
    # scan the package directory to enumerate all .py files under `surge`.
    pkg_dir = Path(__file__).resolve().parent

    results = {}

    # First, scan the package directory for .py files and derive module names
    py_files = [p for p in pkg_dir.iterdir() if p.suffix == '.py']
    file_module_names = [f"surge.{p.stem}" for p in py_files]

    # Merge requested module_names if given (normalize short names to surge.<name>)
    if module_names:
        # ensure unique, preserve order
        for m in module_names:
            if not isinstance(m, str):
                continue
            full = m if m.startswith('surge.') else f"surge.{m}"
            if full not in file_module_names:
                file_module_names.append(full)

    # Normalize module_names for verbose filtering: allow short names like 'metrics'
    # to match 'surge.metrics'. If module_names provided, build a set of normalized
    # full module names to control which modules get the verbose details printed.
    normalized_targets = set()
    if module_names:
        for m in module_names:
            if isinstance(m, str):
                if m.startswith('surge.'):
                    normalized_targets.add(m)
                else:
                    normalized_targets.add(f"surge.{m}")

    for mod_name in file_module_names:
        try:
            mod = importlib.import_module(mod_name)
            loaded = True
            tb = None
        except Exception:
            mod = None
            loaded = False
            tb = traceback.format_exc()

        info = {'loaded': loaded, 'traceback': tb}

        if loaded and mod is not None:
            # List functions defined in this module (not imported)
            all_funcs = inspect.getmembers(mod, inspect.isfunction)
            funcs_defined_here = [name for name, fn in all_funcs if getattr(fn, '__module__', '') == mod.__name__]

            # List classes defined here
            all_classes = inspect.getmembers(mod, inspect.isclass)
            classes_defined_here = [name for name, cls in all_classes if getattr(cls, '__module__', '') == mod.__name__]

            # For convenience also show top-level public names
            public = [a for a in dir(mod) if not a.startswith('_')]

            info.update({
                'functions_list': funcs_defined_here,
                'functions_count': len(funcs_defined_here),
                'classes_list': classes_defined_here,
                'classes_count': len(classes_defined_here),
                'top_public': public[:12],
            })

            # Print according to verbosity: verbose=0 -> concise per module
            # If the caller provided module_names, only print detailed verbose
            # blocks for those modules (helps target a single module for debug).
            should_print_verbose = bool(verbose) and (not normalized_targets or mod_name in normalized_targets)
            if should_print_verbose:
                # Nicely formatted verbose block for each module
                import textwrap

                def wrap(s, width=78, indent=4):
                    return textwrap.fill(s, width=width, subsequent_indent=' ' * indent)

                header = f"Module: {mod_name}"
                print('\n' + header)
                print('-' * len(header))

                # Summary short line
                short = f"{len(classes_defined_here)} classes, {len(funcs_defined_here)} functions"
                print('  Summary: ' + short)

                # Functions (truncate long lists but show count)
                if funcs_defined_here:
                    funcs_str = ', '.join(funcs_defined_here)
                    print('  Functions ({0}):'.format(len(funcs_defined_here)))
                    print('    ' + wrap(funcs_str, indent=6))
                else:
                    print('  Functions (0): -')

                # Classes (truncate/wrap)
                if classes_defined_here:
                    cls_str = ', '.join(classes_defined_here)
                    print('  Classes ({0}):'.format(len(classes_defined_here)))
                    print('    ' + wrap(cls_str, indent=6))
                else:
                    print('  Classes (0): -')

                # Top public names (short)
                if info.get('top_public'):
                    tp = ', '.join(info.get('top_public', []))
                    print('  Top public:')
                    print('    ' + wrap(tp, indent=6))

                # Blank line after module block
                print()

        results[mod_name] = info

    if verbose:
        print('inspect_surge: scan complete')

    # Print a neat ASCII table per module (always print the compact summary).
    # For verbose>=1 this will appear after the detailed per-module output.
    # For verbose==0 this is the only printed output.
    if True:
        # Collect rows
        rows = []
        for mod_name, info in results.items():
            loaded = bool(info.get('loaded', False))
            cls = info.get('classes_count', None)
            funcs = info.get('functions_count', None)
            cls_display = str(cls) if cls is not None else '?'
            funcs_display = str(funcs) if funcs is not None else '?'
            status = 'OK' if loaded else 'FAILED'
            rows.append((mod_name, status, cls_display, funcs_display))

        # Compute column widths
        col_mod = max(len(r[0]) for r in rows) if rows else 10
        col_status = max(len('Status'), max(len(r[1]) for r in rows))
        col_cls = max(len('Classes'), max(len(r[2]) for r in rows))
        col_funcs = max(len('Functions'), max(len(r[3]) for r in rows))

        # Header
        header = f"{ 'Module'.ljust(col_mod) }  { 'Status'.ljust(col_status) }  { 'Classes'.rjust(col_cls) }  { 'Functions'.rjust(col_funcs) }"
        sep = '-' * len(header)
        print(header)
        print(sep)

        # Rows
        for mod_name, status, cls_display, funcs_display in rows:
            print(f"{mod_name.ljust(col_mod)}  {status.ljust(col_status)}  {cls_display.rjust(col_cls)}  {funcs_display.rjust(col_funcs)}")

        print(sep)

    return results


def resolve_device(requested: "str | None" = None):
    """Resolve the torch device for training/inference.

    Priority: explicit ``requested`` > ``SURGE_DEVICE`` env var > default.
    The default is CUDA when available, else CPU — identical results to
    historical behaviour. Apple-Silicon MPS is OPT-IN: pass ``"auto"`` (or
    set ``SURGE_DEVICE=auto``) to pick cuda > mps > cpu, or request
    ``"mps"`` explicitly. Measured on this class of hardware: FNO-2D ~4x
    and U-Net ~2x faster on MPS with identical accuracy, but recurrent
    models (LSTM/GRU) train BADLY on MPS (lorenz63 R2 drops 0.99 -> 0.32),
    which is why MPS is not the silent default.

    ``PYTORCH_ENABLE_MPS_FALLBACK=1`` is set when MPS is chosen so ops
    without MPS kernels fall back to CPU per-op instead of erroring. For
    bit-reproducible runs pin ``SURGE_DEVICE=cpu``.
    """
    import os

    import torch

    req = requested or os.environ.get("SURGE_DEVICE") or None
    auto = req == "auto" or req == ""
    if req is not None and not auto:
        if str(req).startswith("mps"):
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if auto:
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            return torch.device("mps")
    return torch.device("cpu")
