#!/usr/bin/env python3
"""
Test script to diagnose issues with saved PyTorch model files.

This script examines the .pt files in the policy directory and attempts to load them
with various methods to identify corruption or compatibility issues.
"""

import os
import glob
import torch
import pickle
import zipfile


def check_file_integrity(file_path):
    """Check basic file integrity."""
    print(f"\n{'='*60}")
    print(f"Testing file: {file_path}")
    print(f"{'='*60}")

    # Check if file exists
    if not os.path.exists(file_path):
        print("❌ File does not exist!")
        return False

    # Check file size
    file_size = os.path.getsize(file_path)
    print(f"📊 File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")

    if file_size == 0:
        print("❌ File is empty!")
        return False
    elif file_size < 1000:
        print("⚠️  File is very small - might be corrupted")

    # Check if it's a valid ZIP file (PyTorch saves as ZIP)
    try:
        with zipfile.ZipFile(file_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            print(f"✅ Valid ZIP archive with {len(file_list)} files:")
            for f in file_list[:10]:  # Show first 10 files
                print(f"   - {f}")
            if len(file_list) > 10:
                print(f"   ... and {len(file_list) - 10} more files")
        return True
    except zipfile.BadZipFile:
        print("❌ Not a valid ZIP file - file is corrupted or not a PyTorch model")
        return False
    except Exception as e:
        print(f"❌ Error reading ZIP: {e}")
        return False


def check_torch_load_methods(file_path):
    """Check different PyTorch loading methods."""
    print(f"\n🔧 Testing PyTorch loading methods...")

    methods = [
        ("Default", lambda: torch.load(file_path, map_location='cpu')),
        ("weights_only=False", lambda: torch.load(file_path, map_location='cpu', weights_only=False)),
        ("pickle_module=pickle", lambda: torch.load(file_path, map_location='cpu', pickle_module=pickle)),
        ("No map_location", lambda: torch.load(file_path)),
    ]

    for method_name, load_func in methods:
        try:
            print(f"\n   Trying {method_name}...")
            model_data = load_func()
            print(f"   ✅ Success! Type: {type(model_data)}")

            if isinstance(model_data, dict):
                print(f"   📋 Dictionary keys: {list(model_data.keys())}")

                # Check for common keys
                if 'actor_critic' in model_data:
                    print(f"   ✅ Found 'actor_critic' key")
                    actor_critic = model_data['actor_critic']
                    if isinstance(actor_critic, dict):
                        print(f"   📋 actor_critic keys: {list(actor_critic.keys())[:10]}...")
                    print(f"   📊 actor_critic type: {type(actor_critic)}")
                else:
                    print(f"   ⚠️  No 'actor_critic' key found")

                # Check for other common keys
                for key in ['model_state_dict', 'state_dict', 'optimizer', 'epoch']:
                    if key in model_data:
                        print(f"   ✅ Found '{key}' key")

            return model_data

        except Exception as e:
            print(f"   ❌ Failed: {e}")

    return None


def analyze_model_structure(model_data):
    """Analyze the structure of loaded model data."""
    if model_data is None:
        return

    print(f"\n🔍 Analyzing model structure...")

    if isinstance(model_data, dict):
        for key, value in model_data.items():
            print(f"\n   Key: '{key}'")
            print(f"   Type: {type(value)}")

            if isinstance(value, dict):
                print(f"   Sub-keys ({len(value)}): {list(value.keys())[:10]}...")

                # If it looks like state dict, analyze tensor shapes
                if 'weight' in str(value.keys()) or 'bias' in str(value.keys()):
                    print("   📊 Tensor shapes:")
                    for sub_key, tensor in list(value.items())[:10]:
                        if hasattr(tensor, 'shape'):
                            print(f"      {sub_key}: {tensor.shape}")
                        else:
                            print(f"      {sub_key}: {type(tensor)}")

            elif hasattr(value, 'shape'):
                print(f"   Shape: {value.shape}")
            elif hasattr(value, '__len__'):
                print(f"   Length: {len(value)}")


def check_policy_directory(policy_dir):
    """Check all .pt files in the policy directory."""
    print(f"🔍 Scanning policy directory: {policy_dir}")

    if not os.path.exists(policy_dir):
        print(f"❌ Policy directory does not exist: {policy_dir}")
        return

    # Find all .pt files
    pt_files = []
    for subdir in ['upstream', 'downstream']:
        subdir_path = os.path.join(policy_dir, subdir)
        if os.path.exists(subdir_path):
            files = glob.glob(os.path.join(subdir_path, "*.pt"))
            pt_files.extend([(f, subdir) for f in files])

    if not pt_files:
        print("❌ No .pt files found!")
        return

    # Sort files by episode number
    def extract_episode_number(file_info):
        file_path, subdir = file_info
        filename = os.path.basename(file_path)
        # Extract episode number from filename like "real-episode-001-..."
        import re
        match = re.search(r'episode-(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0

    pt_files.sort(key=extract_episode_number)

    print(f"📁 Found {len(pt_files)} .pt files (sorted by episode number):")
    for file_path, subdir in pt_files:
        episode_num = extract_episode_number((file_path, subdir))
        print(f"   - Episode {episode_num:03d} ({subdir}): {os.path.basename(file_path)}")

    print(f"\n{'='*80}")
    print("🧪 TESTING ALL MODELS - CORRUPTION ANALYSIS")
    print(f"{'='*80}")

    # Check each file and track corruption pattern
    successful_loads = []
    failed_loads = []

    for i, (file_path, subdir) in enumerate(pt_files):
        episode_num = extract_episode_number((file_path, subdir))
        print(f"\n📝 [{i+1}/{len(pt_files)}] Episode {episode_num:03d} ({subdir})")
        print(f"File: {os.path.basename(file_path)}")
        print("-" * 60)

        # Check file integrity first
        is_valid_zip = check_file_integrity(file_path)

        if is_valid_zip:
            # Try to load with PyTorch
            try:
                model_data = torch.load(file_path, map_location='cpu', weights_only=False)
                print(f"   ✅ PyTorch load: SUCCESS")
                if isinstance(model_data, dict):
                    print(f"   📋 Keys: {list(model_data.keys())}")
                    if 'actor_critic' in model_data:
                        print(f"   ✅ Has 'actor_critic' key")
                successful_loads.append((episode_num, file_path, subdir))
            except Exception as e:
                print(f"   ❌ PyTorch load: FAILED - {e}")
                failed_loads.append((episode_num, file_path, subdir, str(e)))
        else:
            print(f"   ❌ File integrity: FAILED")
            failed_loads.append((episode_num, file_path, subdir, "File integrity check failed"))

    # Summary analysis
    print(f"\n{'='*80}")
    print("📊 CORRUPTION ANALYSIS SUMMARY")
    print(f"{'='*80}")

    print(f"✅ Successful loads: {len(successful_loads)}/{len(pt_files)}")
    print(f"❌ Failed loads: {len(failed_loads)}/{len(pt_files)}")

    if successful_loads:
        print(f"\n✅ Successfully loaded episodes:")
        for episode_num, file_path, subdir in successful_loads:
            print(f"   - Episode {episode_num:03d} ({subdir})")

    if failed_loads:
        print(f"\n❌ Failed to load episodes:")
        for episode_num, file_path, subdir, error in failed_loads:
            print(f"   - Episode {episode_num:03d} ({subdir}): {error}")

        # Analyze corruption pattern
        failed_episodes = [episode_num for episode_num, _, _, _ in failed_loads]
        successful_episodes = [episode_num for episode_num, _, _ in successful_loads]

        if failed_episodes and successful_episodes:
            first_corruption = min(failed_episodes)
            last_success = max([ep for ep in successful_episodes if ep < first_corruption]) if any(ep < first_corruption for ep in successful_episodes) else None

            print(f"\n🔍 CORRUPTION PATTERN ANALYSIS:")
            if last_success is not None:
                print(f"   📈 Last successful episode: {last_success}")
                print(f"   💥 First corrupted episode: {first_corruption}")
                print(f"   📊 Corruption started after episode {last_success}")
            else:
                print(f"   💥 Corruption present from episode {first_corruption}")

            # Check if corruption is continuous or intermittent
            all_episodes = sorted(set(failed_episodes + successful_episodes))
            corruption_ranges = []
            current_range = []

            for ep in all_episodes:
                if ep in failed_episodes:
                    current_range.append(ep)
                else:
                    if current_range:
                        corruption_ranges.append((min(current_range), max(current_range)))
                        current_range = []
            if current_range:
                corruption_ranges.append((min(current_range), max(current_range)))

            if len(corruption_ranges) == 1:
                start, end = corruption_ranges[0]
                if start == end:
                    print(f"   📍 Single corrupted episode: {start}")
                else:
                    print(f"   📊 Continuous corruption: episodes {start} to {end}")
            elif len(corruption_ranges) > 1:
                print(f"   📊 Intermittent corruption in {len(corruption_ranges)} ranges:")
                for start, end in corruption_ranges:
                    if start == end:
                        print(f"      - Episode {start}")
                    else:
                        print(f"      - Episodes {start} to {end}")

    print(f"\n{'='*80}")


def check_pytorch_version():
    """Check PyTorch version compatibility."""
    print(f"🐍 Python version: {torch.__version__}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"🖥️  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🖥️  CUDA version: {torch.version.cuda}")


def main():
    """Main function to run all tests."""
    import sys

    print("🧪 PyTorch Model Loading Diagnostic Tool")
    print("="*60)

    # Check PyTorch version
    check_pytorch_version()

    # Allow policy directory to be specified as command line argument
    if len(sys.argv) > 1:
        policy_dir = sys.argv[1]
        print(f"\n📁 Using policy directory from command line: {policy_dir}")
    else:
        # Try to find policy directory automatically
        possible_dirs = [
            "/home/edison/Research/omnisafe_zjy/examples/models/torch_save/wabash_0630",
            "examples/models/torch_save/wabash_0630",
            "./examples/models/torch_save/wabash_0630",
        ]

        policy_dir = None
        for dir_path in possible_dirs:
            if os.path.exists(dir_path):
                policy_dir = dir_path
                print(f"\n📁 Found policy directory: {policy_dir}")
                break

        if policy_dir is None:
            print(f"\n❌ Could not find policy directory. Tried:")
            for dir_path in possible_dirs:
                print(f"   - {dir_path}")
            print(f"\nUsage: python {sys.argv[0]} <policy_directory>")
            print(f"Example: python {sys.argv[0]} examples/models/torch_save/wabash_0630")
            return

    # Check policy directory
    check_policy_directory(policy_dir)

    print("\n🏁 Diagnostic complete!")
    print("\nRecommendations:")
    print("1. If files are corrupted: Re-save models from training environment")
    print("2. If version mismatch: Try loading with same PyTorch version used for saving")
    print("3. If structure is different: Update loading code to match actual structure")


if __name__ == "__main__":
    main()
