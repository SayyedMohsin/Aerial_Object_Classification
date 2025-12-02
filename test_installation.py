import sys

def test_imports():
    print("Testing package installations...")
    
    packages = [
        'matplotlib', 'seaborn', 'numpy', 'pandas', 'PIL',
        'tensorflow', 'streamlit', 'cv2', 'sklearn'
    ]
    
    for package in packages:
        try:
            if package == 'PIL':
                import PIL
                print(f"✅ {package} installed successfully")
            elif package == 'cv2':
                import cv2
                print(f"✅ {package} installed successfully")
            elif package == 'sklearn':
                import sklearn
                print(f"✅ {package} installed successfully")
            else:
                __import__(package)
                print(f"✅ {package} installed successfully")
        except ImportError as e:
            print(f"❌ {package} not installed: {e}")

if __name__ == "__main__":
    test_imports()
    input("\nPress Enter to exit...")