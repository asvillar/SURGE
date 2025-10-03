import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import torch

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


def get_data_path(dataset_name="example-data"):
    """
    Get the path to a specific dataset in the SURGE repository.

    Args:
        dataset_name (str): Name of the dataset directory

    Returns:
        Path: Path to the dataset directory
    """
    surge_path = setup_surge_path()
    data_path = surge_path / "data" / "datasets" / dataset_name

    if not data_path.exists():
        available_datasets = []
        datasets_dir = surge_path / "data" / "datasets"
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
    try:
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
        
    except ImportError:
        print(f"\n🔥 PyTorch: Not installed")
        device = "cpu"
    
    # Check sklearn n_jobs usage
    print(f"\n⚙️ Current Model Configuration:")
    print(f"   Random Forest: Using CPU (sklearn)")
    print(f"   n_jobs: -1 (all available cores: {cpu_count_logical})")
    print(f"   Training Device: CPU")
    
    return {
        'cpu_cores_physical': cpu_count_physical,
        'cpu_cores_logical': cpu_count_logical,
        'total_ram_gb': memory.total / (1024**3),
        'gpu_available': gpu_available,
        'gpu_info': gpu_info,
        'device': device if 'device' in locals() else 'cpu'
    }


def enhanced_cpu_monitoring():
    """Enhanced CPU monitoring to capture actual usage during training"""
    
    print("\n🔍 REAL-TIME CPU MONITORING TEST")
    print("=" * 60)
    
    if psutil is None:
        print("⚠️ Warning: psutil not available. Cannot monitor CPU usage.")
        return False
    
    # Baseline CPU usage
    baseline_cpu = psutil.cpu_percent(interval=1)
    print(f"Baseline CPU usage: {baseline_cpu:.1f}%")
    
    # Per-core CPU usage
    per_core_cpu = psutil.cpu_percent(interval=1, percpu=True)
    print(f"Per-core usage: {[f'{cpu:.1f}%' for cpu in per_core_cpu]}")
    
    # Test with a brief computation
    print("\nTesting CPU monitoring during computation...")
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    
    # Create dummy data for CPU test
    X_test = np.random.random((1000, 10))
    y_test = np.random.random(1000)
    
    # Monitor CPU during training
    start_time = time.time()
    model = RandomForestRegressor(n_estimators=50, n_jobs=-1, random_state=42)
    
    # Get CPU usage during fit
    pre_fit_cpu = psutil.cpu_percent()
    model.fit(X_test, y_test)
    post_fit_cpu = psutil.cpu_percent(interval=0.5)  # 0.5 second interval
    
    training_time = time.time() - start_time
    
    print(f"CPU usage: {pre_fit_cpu:.1f}% → {post_fit_cpu:.1f}%")
    print(f"Training time: {training_time:.3f} seconds")
    print(f"CPU utilization detected: {'Yes' if post_fit_cpu > pre_fit_cpu + 5 else 'Low/Variable'}")
    
    return post_fit_cpu > baseline_cpu


def run_compute_resource_analysis():
    """Run complete enhanced compute resource analysis"""
    
    print("🚀 ENHANCED COMPUTE RESOURCE ANALYSIS")
    print("=" * 80)

    system_info = detect_compute_resources()
    cpu_detected = enhanced_cpu_monitoring()

    print(f"\n📊 SUMMARY:")
    print(f"   Training Method: CPU-based (sklearn RandomForest)")
    print(f"   Available Cores: {system_info['cpu_cores_logical']} logical")
    print(f"   RAM Available: {system_info['total_ram_gb']:.1f} GB")
    print(f"   GPU Available: {system_info['gpu_available']}")
    print(f"   CPU Monitoring: {'Working' if cpu_detected else 'Requires longer training for detection'}")

    print(f"\n💡 RECOMMENDATIONS:")
    if system_info['gpu_available'] and 'device' in system_info and system_info['device'] != 'cpu':
        print(f"   ✅ GPU detected - consider PyTorch models for GPU acceleration")
    else:
        print(f"   ✅ CPU training optimal for RandomForest (sklearn)")
    print(f"   ✅ Use n_jobs=-1 to utilize all {system_info['cpu_cores_logical']} cores")
    print(f"   ✅ Current memory is sufficient for large models")

    print("=" * 80)
    
    return system_info, cpu_detected


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

    # Merge requested module_names if given
    if module_names:
        # ensure unique, preserve order
        for m in module_names:
            if m not in file_module_names:
                file_module_names.append(m)

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
            if verbose:
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
