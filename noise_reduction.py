#!/usr/bin/env python3
"""
Audio Noise Reduction using Deep Learning
Converted from Jupyter notebook to standalone Python script
"""
#%%
import librosa
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import soundfile as sf
import matplotlib.pyplot as plt

#%%
# Paths to audio files
clean_path = "/Users/samcourtney/Downloads/DNS-Challenge/clean_output"
noisy_path = "/Users/samcourtney/Downloads/DNS-Challenge/noisy_output"

# Load audio files
clean_files = [os.path.join(clean_path, f) for f in os.listdir(clean_path)]
noisy_files = [os.path.join(noisy_path, f) for f in os.listdir(noisy_path)]

# Load one pair to test
clean_audio, sr = librosa.load(clean_files[0], sr=16000)  # 16kHz sample rate
noisy_audio, _ = librosa.load(noisy_files[0], sr=16000)

print(f"Found {len(clean_files)} clean files and {len(noisy_files)} noisy files")
print(f"Clean path exists: {os.path.exists(clean_path)}")
print(f"Noisy path exists: {os.path.exists(noisy_path)}")
print(f"Sample rate: {sr} Hz")
print(f"Audio length: {len(clean_audio)} samples")

#%%
# Compute STFT
n_fft = 512
hop_length = 128
clean_spec = np.abs(librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length))
noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length))

# Normalize spectrograms by maximum only (preserve magnitude scale)
clean_spec = clean_spec / (np.max(clean_spec) + 1e-10)
noisy_spec = noisy_spec / (np.max(noisy_spec) + 1e-10)

# Check shapes
print(f"Spectrogram shapes - Clean: {clean_spec.shape}, Noisy: {noisy_spec.shape}")

#%%
# Normalization of spectrograms for training
print("Processing all audio files for training...")
X_train, y_train = [], []

for clean_file, noisy_file in zip(clean_files, noisy_files):
    clean_audio, _ = librosa.load(clean_file, sr=16000)
    noisy_audio, _ = librosa.load(noisy_file, sr=16000)
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

#%%
# Split into train (80%), validation (10%), test (10%)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

# Check shapes
print(f"Split shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

#%%
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

#%%
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

# Save model
model.save('noise_reduction_model.keras')
print("Model saved as 'noise_reduction_model.keras'")

#%%
# Evaluate model
print("Evaluating model...")
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

# Make prediction
pred = model.predict(X_test[:1])
print(f"Prediction stats - min: {pred[0].min():.6f}, max: {pred[0].max():.6f}, mean: {pred[0].mean():.6f}")

#%%
# Convert predicted spectrogram to audio with proper phase information
print("Converting prediction to audio...")

# Get the original noisy audio for phase information
noisy_audio_for_phase, _ = librosa.load(noisy_files[0], sr=16000)

# Get complex STFT (with phase) from original noisy audio
noisy_stft = librosa.stft(noisy_audio_for_phase, hop_length=128, n_fft=512)
noisy_phase = np.angle(noisy_stft)

# Use predicted magnitude with original phase
pred_spec = pred[0][:, :, 0]
pred_spec = pred_spec * np.max(np.abs(noisy_stft))  # Restore original scale
pred_complex = pred_spec * np.exp(1j * noisy_phase)  # Combine magnitude and phase
pred_audio = librosa.istft(pred_complex, hop_length=128, n_fft=512)
sf.write('pred_audio.wav', pred_audio, 16000)

# For comparison, also reconstruct test audio properly
test_spec = X_test[0][:, :, 0] * np.max(np.abs(noisy_stft))
test_complex = test_spec * np.exp(1j * noisy_phase)
test_audio = librosa.istft(test_complex, hop_length=128, n_fft=512)
sf.write('test_audio.wav', test_audio, 16000)

#%%
# Test audio amplitude levels
print("Testing audio reconstruction...")

# Load original audio and check levels
clean_audio, sr = librosa.load(clean_files[0], sr=16000)
print(f"Original clean audio - min: {clean_audio.min():.6f}, max: {clean_audio.max():.6f}, RMS: {np.sqrt(np.mean(clean_audio**2)):.6f}")

# Test simple spectrogram -> audio conversion
clean_stft = librosa.stft(clean_audio, n_fft=512, hop_length=128)
reconstructed_audio = librosa.istft(clean_stft, hop_length=128, n_fft=512)
print(f"Simple reconstruction - min: {reconstructed_audio.min():.6f}, max: {reconstructed_audio.max():.6f}, RMS: {np.sqrt(np.mean(reconstructed_audio**2)):.6f}")

# Save both for comparison
sf.write('original_clean.wav', clean_audio, 16000)
sf.write('simple_reconstruction.wav', reconstructed_audio, 16000)

# Now test with normalization/denormalization
clean_spec = np.abs(clean_stft)
original_max = np.max(clean_spec)
normalized_spec = clean_spec / (original_max + 1e-10)
denormalized_spec = normalized_spec * original_max
clean_phase = np.angle(clean_stft)
final_complex = denormalized_spec * np.exp(1j * clean_phase)
final_audio = librosa.istft(final_complex, hop_length=128, n_fft=512)
print(f"Normalized reconstruction - min: {final_audio.min():.6f}, max: {final_audio.max():.6f}, RMS: {np.sqrt(np.mean(final_audio**2)):.6f}")
sf.write('normalized_reconstruction.wav', final_audio, 16000)

#%%
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
