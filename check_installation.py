import sys
import subprocess
import os

def check_installation():
    print("🔍 Checking Installation...")
    
    # Check Python version
    print(f"Python version: {sys.version}")
    
    # Try to import TensorFlow
    try:
        import tensorflow as tf
        print(f"✅ TensorFlow version: {tf.__version__}")
    except ImportError as e:
        print(f"❌ TensorFlow not installed: {e}")
        print("\n🔧 Installing TensorFlow...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow==2.13.0", "--user"])
            print("✅ TensorFlow installed successfully!")
        except Exception as install_error:
            print(f"❌ Installation failed: {install_error}")
    
    # Check other packages
    packages = ['numpy', 'matplotlib', 'PIL', 'streamlit']
    for package in packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package} installed")
            else:
                __import__(package)
                print(f"✅ {package} installed")
        except ImportError:
            print(f"❌ {package} not installed")

if __name__ == "__main__":
    check_installation()
    input("\nPress Enter to exit...")