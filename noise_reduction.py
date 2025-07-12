#!/usr/bin/env python3
"""
Audio Noise Reduction using Deep Learning
Converted from Jupyter notebook to standalone Python script
"""
#%% Cell 1: Imports
import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import soundfile as sf
import matplotlib.pyplot as plt

#%% Cell 2: Load data paths
# Paths to audio files
clean_path = "/Users/samcourtney/Downloads/DNS-Challenge/clean_output"
noisy_path = "/Users/samcourtney/Downloads/DNS-Challenge/noisy_output"

# Load audio files
clean_files = [os.path.join(clean_path, f) for f in os.listdir(clean_path)]
noisy_files = [os.path.join(noisy_path, f) for f in os.listdir(noisy_path)]

print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
print(f"Clean path exists: {os.path.exists(clean_path)}")
print(f"Noisy path exists: {os.path.exists(noisy_path)}")

#%% Cell 3: STFT setup and basic processing
# STFT parameters
n_fft = 512
hop_length = 128
sr = 16000

# Load one pair to test
clean_audio, _ = librosa.load(clean_files[0], sr=sr)
noisy_audio, _ = librosa.load(noisy_files[0], sr=sr)

print(f"Sample rate: {sr} Hz")
print(f"Audio length: {len(clean_audio)} samples")

# Compute STFT for test pair
clean_spec = np.abs(librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length))
noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length))

# Normalize spectrograms by maximum only (preserve magnitude scale)
clean_spec = clean_spec / (np.max(clean_spec) + 1e-10)
noisy_spec = noisy_spec / (np.max(noisy_spec) + 1e-10)

print(f"Spectrogram shapes - Clean: {clean_spec.shape}, Noisy: {noisy_spec.shape}")

#%% Cell 4: Preprocessing demo (hear what training data sounds like)
# Test amplitude effects of preprocessing (hear what training data sounds like)
print("Testing amplitude effects of preprocessing...")

# Original audio
print(f"Original clean audio - min: {clean_audio.min():.6f}, max: {clean_audio.max():.6f}, RMS: {np.sqrt(np.mean(clean_audio**2)):.6f}")

# Simple reconstruction (no normalization)
clean_stft = librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length)
reconstructed_audio = librosa.istft(clean_stft, hop_length=hop_length, n_fft=n_fft)
print(f"Simple reconstruction - min: {reconstructed_audio.min():.6f}, max: {reconstructed_audio.max():.6f}, RMS: {np.sqrt(np.mean(reconstructed_audio**2)):.6f}")

# Normalized reconstruction (with preprocessing - this is what the model sees)
clean_spec = np.abs(clean_stft)
original_max = np.max(clean_spec)
normalized_spec = clean_spec / (original_max + 1e-10)
denormalized_spec = normalized_spec * original_max
clean_phase = np.angle(clean_stft)
final_complex = denormalized_spec * np.exp(1j * clean_phase)
final_audio = librosa.istft(final_complex, hop_length=hop_length, n_fft=n_fft)
print(f"Normalized reconstruction - min: {final_audio.min():.6f}, max: {final_audio.max():.6f}, RMS: {np.sqrt(np.mean(final_audio**2)):.6f}")

# Save comparison files so you can hear the difference
sf.write('original_clean.wav', clean_audio, sr)
sf.write('simple_reconstruction.wav', reconstructed_audio, sr)
sf.write('normalized_reconstruction.wav', final_audio, sr)
print("Saved audio files for comparison:")
print("- original_clean.wav (original)")
print("- simple_reconstruction.wav (STFT round-trip)")
print("- normalized_reconstruction.wav (with preprocessing)")

#%% Cell 5: Process all data for training
# Process all audio files for training
print("Processing all audio files for training...")
X_train, y_train = [], []

for clean_file, noisy_file in zip(clean_files, noisy_files):
    clean_audio, _ = librosa.load(clean_file, sr=sr)
    noisy_audio, _ = librosa.load(noisy_file, sr=sr)
    clean_spec = np.abs(librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length))
    noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length))
    # Scale to max 1.0 without shifting min
    clean_spec = clean_spec / (np.max(clean_spec) + 1e-10)
    noisy_spec = noisy_spec / (np.max(noisy_spec) + 1e-10)
    X_train.append(noisy_spec[..., np.newaxis])
    y_train.append(clean_spec[..., np.newaxis])

X_train = np.array(X_train)
y_train = np.array(y_train)
print(f"Training data shapes - X: {X_train.shape}, y: {y_train.shape}")
print(f"X_train - min: {X_train.min():.6f}, max: {X_train.max():.6f}, mean: {X_train.mean():.6f}")
print(f"y_train - min: {y_train.min():.6f}, max: {y_train.max():.6f}, mean: {y_train.mean():.6f}")

#%% Cell 6: Split data
# Split into train (80%), validation (10%), test (10%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

print(f"Split shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#%% Cell 7: Build model
# Build model
print("Building model...")
model = models.Sequential([
    layers.Input(shape=(257, 1251, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'),
    layers.Cropping2D(((1, 2), (0, 1)))  # Adjusted to get 257x1251
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mse')
model.summary()

#%% Cell 8: Train model
# Train model
print("Training model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=16,
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)

model.save('noise_reduction_model.keras')
print("Model saved as 'noise_reduction_model.keras'")

#%% Cell 9: Evaluate and predict
# Evaluate model and make prediction
print("Evaluating model...")
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

pred = model.predict(X_test[:1])
print(f"Prediction stats - min: {pred[0].min():.6f}, max: {pred[0].max():.6f}, mean: {pred[0].mean():.6f}")

#%% Cell 10: Convert to audio
# Convert prediction to audio
print("Converting prediction to audio...")

# Get the actual test file for proper phase information
test_file_idx = 0  # First test file
test_clean_file = clean_files[test_file_idx]
test_noisy_file = noisy_files[test_file_idx]

# Load the actual test files
test_clean_audio, _ = librosa.load(test_clean_file, sr=sr)
test_noisy_audio, _ = librosa.load(test_noisy_file, sr=sr)

# Get proper phase from the actual test noisy audio
test_noisy_stft = librosa.stft(test_noisy_audio, hop_length=hop_length, n_fft=n_fft)
test_noisy_phase = np.angle(test_noisy_stft)

# Convert prediction to audio using correct phase
pred_spec = pred[0][:, :, 0]
pred_spec = pred_spec * np.max(np.abs(test_noisy_stft))  # Restore original scale
pred_complex = pred_spec * np.exp(1j * test_noisy_phase)  # Combine magnitude and phase
pred_audio = librosa.istft(pred_complex, hop_length=hop_length, n_fft=n_fft)
sf.write('pred_audio.wav', pred_audio, sr)

# Convert test input to audio using correct phase
test_spec = X_test[0][:, :, 0] * np.max(np.abs(test_noisy_stft))
test_complex = test_spec * np.exp(1j * test_noisy_phase)
test_audio = librosa.istft(test_complex, hop_length=hop_length, n_fft=n_fft)
sf.write('test_audio.wav', test_audio, sr)

# Also save the original test files for comparison
sf.write('test_original_clean.wav', test_clean_audio, sr)
sf.write('test_original_noisy.wav', test_noisy_audio, sr)

print("Generated audio files:")
print("- pred_audio.wav (model prediction)")
print("- test_audio.wav (test input)")

#%% Cell 11: Plot training history
# Plot training history
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('training_history.png')
plt.show()

print("Pipeline completed successfully!")
print("Generated files:")
print("- noise_reduction_model.keras (trained model)")
print("- pred_audio.wav (predicted clean audio)")
print("- test_audio.wav (test noisy audio)")
print("- original_clean.wav (original clean audio)")
print("- simple_reconstruction.wav (simple reconstruction)")
print("- normalized_reconstruction.wav (normalized reconstruction)")
print("- training_history.png (training plot)")
