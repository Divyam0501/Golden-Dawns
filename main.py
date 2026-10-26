import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import hashlib
import sounddevice as sd
import soundfile as sf

def record_audio(output_folder="recordings", duration=5, sr=22050):
    """Records audio from the microphone and saves it to a WAV file."""
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "hummed_tune.wav")
    
    print(f"\n🎤 Get ready... Recording for {duration} seconds. Start humming/singing!")
    print("3... 2... 1... GO!")
    
    # Record audio from the microphone
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    
    # Save to file
    sf.write(file_path, audio_data, sr)
    print(f"✅ Recording saved successfully at: {file_path}")
    
    return file_path

def process_audio_phase1(audio_path, sr=22050):
    print(f"\nLoading audio file: {audio_path}")
    
    # 1. Load Audio Waveform
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # ==========================================
    # VISUALIZATION 1: BEFORE GRAPHS
    # ==========================================
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
    
    # 1a. Raw Waveform (Time Domain)
    librosa.display.waveshow(y, sr=sr, ax=axes[0], color='royalblue')
    axes[0].set_title('Before: Raw Audio Waveform (Time Domain)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    
    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # 1b. Full Spectrogram (Frequency Domain / Fourier Series Representation)
    img = librosa.display.specshow(stft_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=axes[1], cmap='coolwarm')
    axes[1].set_title('Before: Full Frequency Spectrogram (STFT)')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Frequency (Hz)')
    fig.colorbar(img, ax=axes[1], format='%+2.0f dB')
    
    plt.tight_layout()
    print("Displaying 'Before' graphs...")
    plt.show()
    
    # ==========================================
    # PROCESSING: PEAK EXTRACTION & HASHING
    # ==========================================
    peak_frequencies = np.argmax(np.abs(stft), axis=0)
    times = librosa.times_like(peak_frequencies, sr=sr, hop_length=512)
    
    # Quantize peaks to make them resilient against slight pitch variations during humming
    quantized_peaks = (peak_frequencies // 4) * 4
    
    # ==========================================
    # VISUALIZATION 2: AFTER GRAPH
    # ==========================================
    fig_after, ax_after = plt.subplots(figsize=(10, 4))
    
    # Plot Peak Diagram / Constellation Map
    ax_after.scatter(times, quantized_peaks, color='crimson', s=12, label='Extracted Peak Constituents')
    ax_after.set_title('After: Peak Diagram / Constellation Map (Audio Fingerprint Features)')
    ax_after.set_xlabel('Time (s)')
    ax_after.set_ylabel('Quantized Frequency Bins')
    ax_after.legend(loc='upper right')
    ax_after.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    print("Displaying 'After' graph...")
    plt.show()
    
    # ==========================================
    # HASHING & HESACODE GENERATION
    # ==========================================
    hash_list = []
    for i in range(len(quantized_peaks) - 1):
        pair = f"{quantized_peaks[i]}-{quantized_peaks[i+1]}"
        sub_hash = hashlib.md5(pair.encode()).hexdigest()[:8]
        hash_list.append(sub_hash)
        
    combined_fingerprint = "".join(hash_list)
    final_hexacode = hashlib.sha256(combined_fingerprint.encode()).hexdigest()
    
    print("\n--- Phase 1 Processing Successful ---")
    print(f"Generated Hexacode: {final_hexacode}")
    
    return final_hexacode, hash_list

if __name__ == "__main__":
    # Step 1: Record from microphone and save to folder
    saved_audio_path = record_audio(output_folder="recordings", duration=5)
    
    # Step 2: Run Phase 1 analysis, graphs, and hexacode conversion
    hexacode, sub_hashes = process_audio_phase1(saved_audio_path)