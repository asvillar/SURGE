#!/usr/bin/env python
"""
SURGE Installation Checker
Verifies that SURGE is properly installed and accessible
"""

import sys
import os
from pathlib import Path

# Ensure SURGE root is on sys.path so imports succeed when running from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent
SURGE_ROOT = SCRIPT_DIR.parent.parent
if str(SURGE_ROOT) not in sys.path:
    sys.path.insert(0, str(SURGE_ROOT))

def check_surge_installation():
    """Check if SURGE is properly installed and print diagnostic information"""
    
    print("=" * 60)
    print("SURGE Installation Checker")
    print("=" * 60)
    print()
    
    # 1. Check if SURGE can be imported
    print("1. Checking SURGE import...")
    try:
        import surge
        print("   ✅ SURGE successfully imported")
        print(f"   📦 Version: {surge.__version__}")
        print(f"   📁 Location: {surge.__file__}")
        surge_imported = True
    except ImportError as e:
        print(f"   ❌ Cannot import SURGE: {e}")
        surge_imported = False
    print()
    
    # 2. Check Python path
    print("2. Python search paths:")
    for i, path in enumerate(sys.path[:5], 1):
        print(f"   {i}. {path}")
    if len(sys.path) > 5:
        print(f"   ... and {len(sys.path) - 5} more paths")
    print()
    
    # 3. Check for setup.py (indicates SURGE root)
    print("3. Checking for SURGE directory...")
    current_dir = Path.cwd()
    surge_root = None
    
    # Check current directory
    if (current_dir / 'setup.py').exists() and (current_dir / 'surge').exists():
        surge_root = current_dir
        print(f"   ✅ SURGE root found: {surge_root}")
    else:
        # Check parent directories
        for parent in current_dir.parents:
            if (parent / 'setup.py').exists() and (parent / 'surge').exists():
                surge_root = parent
                print(f"   ✅ SURGE root found: {surge_root}")
                break
        if not surge_root:
            print(f"   ⚠️  SURGE root not found in current path")
            print(f"   📂 Current directory: {current_dir}")
    print()
    
    # 4. Check conda environment
    print("4. Checking Python environment...")
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'Not using conda')
    print(f"   🐍 Python: {sys.version.split()[0]}")
    print(f"   📦 Conda env: {conda_env}")
    print(f"   📍 Python executable: {sys.executable}")
    print()
    
    # 5. Check SURGE components if imported
    if surge_imported:
        print("5. Checking SURGE components...")
        try:
            from surge.trainer import MLTrainer
            print("   ✅ MLTrainer available")
        except ImportError:
            print("   ❌ MLTrainer not available")
        
        try:
            from surge import PYTORCH_AVAILABLE, GPFLOW_AVAILABLE
            print(f"   {'✅' if PYTORCH_AVAILABLE else '❌'} PyTorch: {PYTORCH_AVAILABLE}")
            print(f"   {'✅' if GPFLOW_AVAILABLE else '❌'} GPflow: {GPFLOW_AVAILABLE}")
        except ImportError:
            print("   ❌ Backend flags not available")
        print()
    
    # 6. Recommendations
    print("=" * 60)
    print("Recommendations:")
    print("=" * 60)
    
    if not surge_imported:
        print("❌ SURGE is not properly installed.")
        print()
        print("To fix:")
        print("  1. Navigate to SURGE directory: cd /path/to/SURGE")
        print("  2. Activate conda environment: conda activate surge")
        print("  3. Install SURGE: pip install -e .")
        print()
    else:
        print("✅ SURGE is properly installed and working!")
        print()
        print("Next steps:")
        print("  • Try examples: python examples/simple_optuna_demo.py")
        print("  • Run tests: pytest tests/ -v")
        print("  • Check docs: docs/setup/INSTALLATION.md")
        print()
    
    # 7. Environment variable suggestion
    if surge_root and not surge_imported:
        print("💡 Quick fix - Add to your shell config (~/.bashrc or ~/.zshrc):")
        print(f"   export PYTHONPATH=\"{surge_root}:$PYTHONPATH\"")
        print(f"   export SURGE_DIR=\"{surge_root}\"")
        print()
        print("   Then reload: source ~/.bashrc")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    check_surge_installation()
