import librosa
import numpy as np
import os
import tensorflow as tf
import soundfile as sf

# Configuration
MODEL_PATH = 'noise_reduction_model_5k.keras'
NOISY_PATH = r"C:\Users\samth\OneDrive\Documents\medium\DNS-Challenge\training_output\noisy"
CLEAN_PATH = r"C:\Users\samth\OneDrive\Documents\medium\DNS-Challenge\training_output\clean"
OUTPUT_PATH = 'test_output'
NUM_TEST_SAMPLES = 5

# Create output directory
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("=" * 60)
print("NOISE REDUCTION MODEL - TESTING")
print("=" * 60)

# Load the trained model
print("\n[LOADING] Loading trained model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"[OK] Model loaded from '{MODEL_PATH}'")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    exit(1)

# Get test files
print("\n[LOADING] Loading test files...")
noisy_files = [os.path.join(NOISY_PATH, f) for f in os.listdir(NOISY_PATH) if f.endswith('.wav')]
clean_files = [os.path.join(CLEAN_PATH, f) for f in os.listdir(CLEAN_PATH) if f.endswith('.wav')]

# Sort to ensure consistent pairing
noisy_files.sort()
clean_files.sort()

# Select test samples (use files not in training set)
# Since we used first 5000 for training, we can use files after that
# Or we can just use the last NUM_TEST_SAMPLES from our dataset
test_noisy_files = noisy_files[-NUM_TEST_SAMPLES:]
test_clean_files = clean_files[-NUM_TEST_SAMPLES:]

print(f"[OK] Selected {NUM_TEST_SAMPLES} test files")

# STFT parameters (same as training) - Updated for 44.1kHz
n_fft = 2048  # Increased from 512 for better frequency resolution at 44.1kHz
hop_length = 512  # Increased from 128 to maintain similar time resolution
target_frames = 861  # Adjusted for 44.1kHz

print("\n" + "=" * 60)
print("PROCESSING TEST FILES")
print("=" * 60)

for idx, (noisy_file, clean_file) in enumerate(zip(test_noisy_files, test_clean_files), 1):
    print(f"\n[{idx}/{NUM_TEST_SAMPLES}] Processing: {os.path.basename(noisy_file)}")

    # Load audio at 44.1kHz
    noisy_audio, sr = librosa.load(noisy_file, sr=44100)
    clean_audio, _ = librosa.load(clean_file, sr=44100)
    original_length = len(noisy_audio)

    # Convert to spectrogram
    noisy_spec = np.abs(librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length))
    clean_spec = np.abs(librosa.stft(clean_audio, n_fft=n_fft, hop_length=hop_length))

    # Get phase for reconstruction
    noisy_complex = librosa.stft(noisy_audio, n_fft=n_fft, hop_length=hop_length)
    phase = np.angle(noisy_complex)

    # Pad or crop to fixed length
    original_frames = noisy_spec.shape[1]
    if noisy_spec.shape[1] < target_frames:
        pad_width = target_frames - noisy_spec.shape[1]
        noisy_spec = np.pad(noisy_spec, ((0, 0), (0, pad_width)), mode='constant')
        phase = np.pad(phase, ((0, 0), (0, pad_width)), mode='constant')
    else:
        noisy_spec = noisy_spec[:, :target_frames]
        phase = phase[:, :target_frames]

    # Normalize
    max_val = np.max(noisy_spec)
    noisy_spec_norm = noisy_spec / (max_val + 1e-10)

    # Prepare for model input
    input_spec = noisy_spec_norm[np.newaxis, ..., np.newaxis]

    # Predict
    print("   [PROCESSING] Running model prediction...")
    predicted_spec = model.predict(input_spec, verbose=0)
    predicted_spec = predicted_spec[0, :, :, 0]

    # Denormalize
    predicted_spec = predicted_spec * max_val

    # Crop back to original length
    predicted_spec = predicted_spec[:, :original_frames]
    phase = phase[:, :original_frames]

    # Reconstruct audio using predicted magnitude and original phase
    predicted_complex = predicted_spec * np.exp(1j * phase)
    denoised_audio = librosa.istft(predicted_complex, hop_length=hop_length, length=original_length)

    # Calculate metrics
    mse_noisy = np.mean((noisy_audio - clean_audio) ** 2)
    mse_denoised = np.mean((denoised_audio - clean_audio) ** 2)
    improvement = ((mse_noisy - mse_denoised) / mse_noisy) * 100

    print(f"   [METRICS]")
    print(f"      MSE (noisy):    {mse_noisy:.6f}")
    print(f"      MSE (denoised): {mse_denoised:.6f}")
    print(f"      Improvement:    {improvement:.2f}%")

    # Save outputs
    base_name = os.path.splitext(os.path.basename(noisy_file))[0]

    noisy_output = os.path.join(OUTPUT_PATH, f"{base_name}_noisy.wav")
    clean_output = os.path.join(OUTPUT_PATH, f"{base_name}_clean.wav")
    denoised_output = os.path.join(OUTPUT_PATH, f"{base_name}_denoised.wav")

    sf.write(noisy_output, noisy_audio, sr)
    sf.write(clean_output, clean_audio, sr)
    sf.write(denoised_output, denoised_audio, sr)

    print(f"   [SAVED]")
    print(f"      Noisy:    {noisy_output}")
    print(f"      Clean:    {clean_output}")
    print(f"      Denoised: {denoised_output}")

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)
print(f"\nResults saved to: {os.path.abspath(OUTPUT_PATH)}")
print("\nYou can now listen to the audio files to compare:")
print("  - *_noisy.wav: Original noisy audio")
print("  - *_clean.wav: Ground truth clean audio")
print("  - *_denoised.wav: Model-denoised audio")
