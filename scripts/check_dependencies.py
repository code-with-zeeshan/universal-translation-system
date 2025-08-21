#!/usr/bin/env python
# scripts/check_dependencies.py
import json
import pkg_resources
import sys
from pathlib import Path
import importlib
import subprocess
import platform

def check_dependencies():
    """Check installed dependencies against requirements"""
    try:
        # Load version config
        version_config_path = Path(__file__).parent.parent / 'version-config.json'
        with open(version_config_path) as f:
            version_config = json.load(f)
        
        print(f"🔍 Checking dependencies for Universal Translation System v{version_config['core']['version']}")
        
        # Check installed packages
        requirements_path = Path(__file__).parent.parent / 'requirements.txt'
        with open(requirements_path) as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        missing = []
        outdated = []
        
        for req in requirements:
            if '>=' in req:
                pkg_name, version_req = req.split('>=', 1)
                pkg_name = pkg_name.strip()
                if '<' in version_req:
                    version_req = version_req.split('<', 1)[0].strip()
            elif '==' in req:
                pkg_name, version_req = req.split('==', 1)
                pkg_name = pkg_name.strip()
            else:
                pkg_name = req.strip()
                version_req = None
            
            try:
                installed = pkg_resources.get_distribution(pkg_name)
                if version_req and pkg_resources.parse_version(installed.version) < pkg_resources.parse_version(version_req):
                    outdated.append(f"{pkg_name} (installed: {installed.version}, required: >={version_req})")
            except pkg_resources.DistributionNotFound:
                missing.append(pkg_name)
        
        # Check critical dependencies
        critical_deps = {
            'torch': 'PyTorch',
            'transformers': 'Transformers',
            'sentencepiece': 'SentencePiece',
            'msgpack': 'MessagePack',
            'numpy': 'NumPy',
            'pyyaml': 'PyYAML',
            'tqdm': 'tqdm'
        }
        
        critical_missing = [critical_deps[pkg] for pkg in critical_deps if pkg in missing]
        
        # Check CUDA availability
        cuda_available = False
        cuda_version = "Not available"
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            if cuda_available:
                cuda_version = torch.version.cuda
        except ImportError:
            pass
        
        # Print results
        print("\n=== System Information ===")
        print(f"Python version: {platform.python_version()}")
        print(f"OS: {platform.system()} {platform.release()}")
        print(f"CUDA available: {'✅ Yes' if cuda_available else '❌ No'}")
        if cuda_available:
            print(f"CUDA version: {cuda_version}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
        
        print("\n=== Dependency Check ===")
        if critical_missing:
            print(f"❌ Critical dependencies missing: {', '.join(critical_missing)}")
            print("   These dependencies are required for the system to function.")
            print(f"   Install them with: pip install {' '.join(critical_missing)}")
        
        if missing:
            print("❌ Missing dependencies:")
            for pkg in missing:
                print(f"  - {pkg}")
            print(f"\nInstall missing dependencies with: pip install {' '.join(missing)}")
        
        if outdated:
            print("\n⚠️ Outdated dependencies:")
            for pkg in outdated:
                print(f"  - {pkg}")
            print("\nUpdate outdated dependencies with: pip install -U <package_name>")
        
        if not missing and not outdated:
            print("✅ All dependencies are installed and up to date!")
        
        # Check for optional components
        print("\n=== Optional Components ===")
        
        # Check Docker
        docker_available = False
        try:
            result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                docker_available = True
                print(f"✅ Docker: {result.stdout.strip()}")
            else:
                print("❌ Docker: Not installed or not in PATH")
        except FileNotFoundError:
            print("❌ Docker: Not installed or not in PATH")
        
        # Check Node.js (for Web/React Native SDKs)
        nodejs_available = False
        try:
            result = subprocess.run(["node", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                nodejs_available = True
                print(f"✅ Node.js: {result.stdout.strip()}")
            else:
                print("❌ Node.js: Not installed or not in PATH")
        except FileNotFoundError:
            print("❌ Node.js: Not installed or not in PATH")
        
        # Check Flutter (for Flutter SDK)
        flutter_available = False
        try:
            result = subprocess.run(["flutter", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                flutter_available = True
                print(f"✅ Flutter: Available")
            else:
                print("❌ Flutter: Not installed or not in PATH")
        except FileNotFoundError:
            print("❌ Flutter: Not installed or not in PATH")
        
        print("\n=== Recommendations ===")
        if not cuda_available:
            print("⚠️ CUDA not available. Training and decoder will run on CPU (very slow).")
            print("   For better performance, install CUDA and a compatible PyTorch version.")
        
        if not docker_available:
            print("⚠️ Docker not installed. Containerized deployment will not be available.")
            print("   Install Docker for easier deployment and testing.")
        
        if not nodejs_available and (Path(__file__).parent.parent / 'web').exists():
            print("⚠️ Node.js not installed. Web and React Native SDK development will not be possible.")
            print("   Install Node.js for Web and React Native SDK development.")
        
        if not flutter_available and (Path(__file__).parent.parent / 'flutter').exists():
            print("⚠️ Flutter not installed. Flutter SDK development will not be possible.")
            print("   Install Flutter for Flutter SDK development.")
        
        return not critical_missing
    
    except Exception as e:
        print(f"Error checking dependencies: {e}")
        return False

if __name__ == "__main__":
    success = check_dependencies()
    sys.exit(0 if success else 1)