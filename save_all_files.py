import os
import shutil

print("💾 SAVING ALL PROJECT FILES...")

# Create backup folder
backup_folder = "project_backup"
if os.path.exists(backup_folder):
    shutil.rmtree(backup_folder)

os.makedirs(backup_folder)

# Files to save
files_to_save = [
    # App files
    ("app_perfect_final.py", "app.py"),
    
    # Source files
    ("src/data_analysis.py", "src/data_analysis.py"),
    ("src/train_classification_fixed.py", "src/train_classification_fixed.py"),
    ("src/debug_training.py", "src/debug_training.py"),
    
    # Test files
    ("test_images.py", "test_images.py"),
    
    # Config files
    ("requirements.txt", "requirements.txt"),
    ("README.md", "README.md"),
    
    # Model files (if exist)
    ("models/proper_model_weights.weights.h5", "models/proper_model_weights.weights.h5"),
    ("models/final_model.h5", "models/final_model.h5"),
]

# Copy files
for src, dst in files_to_save:
    src_path = src
    dst_path = os.path.join(backup_folder, dst)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        print(f"✅ Saved: {dst}")
    else:
        print(f"⚠️ Not found: {src}")

print(f"\n🎉 All files saved to: {backup_folder}/")
print("\n📁 Folder structure:")
for root, dirs, files in os.walk(backup_folder):
    level = root.replace(backup_folder, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for file in files:
        print(f"{subindent}{file}")