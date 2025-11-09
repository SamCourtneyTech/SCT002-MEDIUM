import os
import re

# Configuration
clean_path = r"C:\Users\samth\OneDrive\Documents\medium\DNS-Challenge\training_output\clean"
noisy_path = r"C:\Users\samth\OneDrive\Documents\medium\DNS-Challenge\training_output\noisy"

print("=" * 60)
print("VERIFYING FILE PAIRING LOGIC")
print("=" * 60)

# Get all clean files
clean_file_list = [f for f in os.listdir(clean_path) if f.endswith('.wav')]
noisy_file_list = [f for f in os.listdir(noisy_path) if f.endswith('.wav')]

print(f"\n[INFO] Total clean files found: {len(clean_file_list)}")
print(f"[INFO] Total noisy files found: {len(noisy_file_list)}")

# Create a mapping of fileid to noisy filename
print("\n[MATCHING] Creating fileid mapping...")
noisy_map = {}
for noisy_file in noisy_file_list:
    match = re.search(r'fileid_(\d+)', noisy_file)
    if match:
        fileid = int(match.group(1))
        noisy_map[fileid] = noisy_file

print(f"[OK] Noisy files mapped: {len(noisy_map)}")

# Match clean files with their corresponding noisy files
matched_pairs = []
for clean_file in clean_file_list:
    match = re.search(r'clean_fileid_(\d+)', clean_file)
    if match:
        fileid = int(match.group(1))
        if fileid in noisy_map:
            matched_pairs.append((clean_file, noisy_map[fileid], fileid))

print(f"[OK] Matched pairs: {len(matched_pairs)}")

# Sort by fileid for consistency
matched_pairs.sort(key=lambda x: x[2])

# Show first 10 matched pairs
print("\n" + "=" * 60)
print("FIRST 10 MATCHED PAIRS")
print("=" * 60)
for i, (clean, noisy, fileid) in enumerate(matched_pairs[:10], 1):
    print(f"\n[{i}] FileID: {fileid}")
    print(f"    Clean: {clean}")
    print(f"    Noisy: {noisy[:90]}...")

# Show last 5 matched pairs
print("\n" + "=" * 60)
print("LAST 5 MATCHED PAIRS")
print("=" * 60)
for i, (clean, noisy, fileid) in enumerate(matched_pairs[-5:], len(matched_pairs)-4):
    print(f"\n[{i}] FileID: {fileid}")
    print(f"    Clean: {clean}")
    print(f"    Noisy: {noisy[:90]}...")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
print(f"\n[SUMMARY]")
print(f"   Clean files: {len(clean_file_list)}")
print(f"   Noisy files: {len(noisy_file_list)}")
print(f"   Matched pairs: {len(matched_pairs)}")
print(f"   Match rate: {len(matched_pairs) / len(clean_file_list) * 100:.2f}% of clean files have matching noisy files")
