import librosa
import numpy as np
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models

# GPU Detection and Configuration
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[OK] {len(gpus)} GPU(s) detected:")
    for gpu in gpus:
        print(f"   - {gpu.name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("[OK] GPU memory growth enabled")
    except RuntimeError as e:
        print(f"[WARNING] GPU configuration warning: {e}")
else:
    print("[WARNING] No GPU detected - training will use CPU (much slower)")
print("=" * 60)

# Configuration
MAX_FILES = 5000
clean_path = r"C:\Users\samth\OneDrive\Documents\medium\DNS-Challenge\training_output\clean"
noisy_path = r"C:\Users\samth\OneDrive\Documents\medium\DNS-Challenge\training_output\noisy"

# Load audio files and match by fileid
print("\n[LOADING] Loading file lists and matching by fileid...")
import re

# Get all clean files
clean_file_list = [f for f in os.listdir(clean_path) if f.endswith('.wav')]
noisy_file_list = [f for f in os.listdir(noisy_path) if f.endswith('.wav')]

print(f"   Total clean files found: {len(clean_file_list)}")
print(f"   Total noisy files found: {len(noisy_file_list)}")

# Create a mapping of fileid to noisy filename
print("\n[MATCHING] Creating fileid mapping...")
noisy_map = {}
for noisy_file in noisy_file_list:
    match = re.search(r'fileid_(\d+)', noisy_file)
    if match:
        fileid = int(match.group(1))
        noisy_map[fileid] = noisy_file

print(f"   Noisy files mapped: {len(noisy_map)}")

# Match clean files with their corresponding noisy files
matched_pairs = []
for clean_file in clean_file_list:
    match = re.search(r'clean_fileid_(\d+)', clean_file)
    if match:
        fileid = int(match.group(1))
        if fileid in noisy_map:
            clean_path_full = os.path.join(clean_path, clean_file)
            noisy_path_full = os.path.join(noisy_path, noisy_map[fileid])
            matched_pairs.append((clean_path_full, noisy_path_full, fileid))

print(f"   Matched pairs: {len(matched_pairs)}")

# Sort by fileid for consistency
matched_pairs.sort(key=lambda x: x[2])

# Limit to MAX_FILES
if MAX_FILES is not None and len(matched_pairs) > MAX_FILES:
    matched_pairs = matched_pairs[:MAX_FILES]
    print(f"   Limited to: {MAX_FILES} pairs")

# Extract clean and noisy file paths
clean_files = [pair[0] for pair in matched_pairs]
noisy_files = [pair[1] for pair in matched_pairs]

print(f"\nDataset Information:")
print(f"   Using {len(matched_pairs)} matched pairs for training")
print(f"   First pair example:")
print(f"      Clean: {os.path.basename(clean_files[0])}")
print(f"      Noisy: {os.path.basename(noisy_files[0])[:80]}...")
print(f"      FileID: {matched_pairs[0][2]}")

# Prepare training data
print("\n[PROCESSING] Processing audio files to spectrograms...")
X_train_list, y_train_list = [], []
# Updated for 44.1kHz sampling rate
n_fft = 2048  # Increased from 512 for better frequency resolution at 44.1kHz
hop_length = 512  # Increased from 128 to maintain similar time resolution
target_frames = 861  # Adjusted for 44.1kHz (10 seconds @ 44.1kHz with hop_length=512)

for idx, (clean_file, noisy_file) in enumerate(zip(clean_files, noisy_files)):
    if (idx + 1) % 500 == 0:
        print(f"   Processed {idx + 1}/{len(clean_files)} files...")

    clean_audio, _ = librosa.load(clean_file, sr=44100)
    noisy_audio, _ = librosa.load(noisy_file, sr=44100)
    clean_spec = np.abs(librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length))
    noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length))

    # Pad or crop to fixed length
    if clean_spec.shape[1] < target_frames:
        # Pad with zeros
        pad_width = target_frames - clean_spec.shape[1]
        clean_spec = np.pad(clean_spec, ((0, 0), (0, pad_width)), mode='constant')
        noisy_spec = np.pad(noisy_spec, ((0, 0), (0, pad_width)), mode='constant')
    else:
        # Crop to target length
        clean_spec = clean_spec[:, :target_frames]
        noisy_spec = noisy_spec[:, :target_frames]

    # Normalize spectrograms
    clean_spec = clean_spec / (np.max(clean_spec) + 1e-10)
    noisy_spec = noisy_spec / (np.max(noisy_spec) + 1e-10)

    X_train_list.append(noisy_spec[..., np.newaxis])
    y_train_list.append(clean_spec[..., np.newaxis])

X_train = np.array(X_train_list)
y_train = np.array(y_train_list)

print(f"[OK] Data prepared:")
print(f"   X_train range: [{X_train.min():.4f}, {X_train.max():.4f}], mean: {X_train.mean():.6f}")
print(f"   y_train range: [{y_train.min():.4f}, {y_train.max():.4f}], mean: {y_train.mean():.6f}")

# Split into train/validation/test
print("\n[SPLITTING] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Validation samples: {X_val.shape[0]}")
print(f"   Test samples: {X_test.shape[0]}")

# Build model
print("\n[BUILDING] Building model...")
# Updated input shape for 44.1kHz: n_fft=2048 -> 1025 freq bins, target_frames=861
model = models.Sequential([
    layers.Input(shape=(1025, 861, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
    layers.Cropping2D(((1, 2), (1, 0)))
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
model.summary()

# Train model
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Epochs: 50 (with early stopping)")
print(f"Batch size: 16")
print(f"Learning rate: 0.0001")
print("=" * 60 + "\n")

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=16,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)

# Save model
model_path = 'noise_reduction_model_5k.keras'
model.save(model_path)
print(f"\n[OK] Model saved to '{model_path}'")

# Evaluate on test set
print("\n[EVALUATING] Evaluating on test set...")
test_loss = model.evaluate(X_test, y_test)
print(f"[OK] Test loss: {test_loss:.6f}")

print("\n[COMPLETE] Training complete!")
