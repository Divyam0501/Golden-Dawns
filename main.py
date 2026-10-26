import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf
from google import genai
from google.genai.errors import ServerError

# ==========================================
# ENTER YOUR GEMINI API KEY HERE
# ==========================================
GEMINI_API_KEY = "AQ.Ab8RN6IZXleHOiDxdWGp0vCYsC5tfwM2eGuX5TKuMX7MmYSJsw"  # Get free from Google AI Studio
# ==========================================

# Initialize Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

def record_audio(output_folder="recordings", duration=10, sr=22050):
    """Records audio from the microphone for 10 seconds and saves it to a WAV file."""
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, "hummed_tune.wav")
    
    print(f"\n🎤 Hum or sing your song clearly for {duration} seconds...")
    print("3... 2... 1... GO!")
    
    audio_data = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    
    sf.write(file_path, audio_data, sr)
    print(f"✅ Recording saved at: {file_path}")
    return file_path

def process_audio_pipeline(audio_path, sr=22050):
    print(f"\nLoading audio file for high-accuracy feature extraction...")
    y, sr = librosa.load(audio_path, sr=sr, mono=True)
    
    # ==========================================================
    # STEP 1: INITIAL AUDIO WAVEFORM (TIME DOMAIN) GRAPH
    # ==========================================================
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr, ax=ax1, color='royalblue')
    ax1.set_title('Step 1: Initial Audio Waveform (Time Domain - 10s Recording)')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    plt.tight_layout()
    print("📈 Displaying Step 1 Graph: Initial Audio Waveform...")
    plt.show()
    
    # Compute Short-Time Fourier Transform (STFT)
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    
    # ==========================================================
    # STEP 2: FOURIER TRANSFORM SPECTROGRAM & PEAK MAP GRAPH
    # ==========================================================
    fig2, axes2 = plt.subplots(nrows=2, ncols=1, figsize=(10, 7))
    
    # 2a. Full Frequency Spectrogram
    img = librosa.display.specshow(stft_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', ax=axes2[0], cmap='coolwarm')
    axes2[0].set_title('Step 2a: Fourier Transform Frequency Spectrogram (STFT)')
    axes2[0].set_xlabel('Time (s)')
    axes2[0].set_ylabel('Frequency (Hz)')
    fig2.colorbar(img, ax=axes2[0], format='%+2.0f dB')
    
    # Extract dominant pitch contours over time
    peak_frequencies = np.argmax(np.abs(stft), axis=0)
    times = librosa.times_like(peak_frequencies, sr=sr, hop_length=512)
    quantized_peaks = (peak_frequencies // 4) * 4
    
    # 2b. Peak Constellation Map
    axes2[1].scatter(times, quantized_peaks, color='crimson', s=10, label='Dominant Harmonic Peaks')
    axes2[1].set_title('Step 2b: Fourier Peak Constellation Map')
    axes2[1].set_xlabel('Time (s)')
    axes2[1].set_ylabel('Quantized Frequency Bins')
    axes2[1].legend(loc='upper right')
    axes2[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    print("📈 Displaying Step 2 Graphs: Fourier Spectrogram & Peak Map...")
    plt.show()
    
    # ==========================================================
    # STEP 3: CHROMA FEATURE EXTRACTOR (PITCH CLASS PROFILE) GRAPH
    # ==========================================================
    # Chroma captures the musical notes (C, C#, D... B) regardless of octave
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=512)
    
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    img_chroma = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', ax=ax3, cmap='coolwarm')
    ax3.set_title('Step 3: Musical Pitch Class Profile (Chroma Features)')
    fig3.colorbar(img_chroma, ax=ax3)
    plt.tight_layout()
    print("📈 Displaying Step 3 Graph: Pitch Chroma Profile...")
    plt.show()
    
    # Generate Melodic Contour sequence (UP, DOWN, FLAT transitions)
    reduced_peaks = peak_frequencies[::12]
    contour = []
    for i in range(len(reduced_peaks) - 1):
        diff = reduced_peaks[i+1] - reduced_peaks[i]
        if diff > 4:
            contour.append("UP")
        elif diff < -4:
            contour.append("DOWN")
        else:
            contour.append("FLAT")
            
    # Compress consecutive directions
    compressed_contour = []
    last_move = None
    for move in contour:
        if move != last_move:
            compressed_contour.append(move)
            last_move = move
            
    contour_signature = " -> ".join(compressed_contour[:40])
    
    # Extract dominant key/pitch classes present in the chroma vector
    mean_chroma = np.mean(chroma, axis=1)
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    active_notes = [note_names[i] for i in np.argsort(mean_chroma)[-4:][::-1]]
    
    print("\n--- Audio Profile Successfully Extracted ---")
    print(f"Dominant Musical Notes: {active_notes}")
    print(f"Extracted Contour Pattern: {contour_signature}")
    
    return contour_signature, active_notes

def predict_song_with_gemma(contour_signature, active_notes):
    """Sends high-accuracy musical attributes directly to Gemma 4 (31B) for song identification."""
    print("\n🤖 Step 4: Sending Pitch & Contour Profile to Gemma 4 (31B) for High-Accuracy Prediction...")

    prompt = f"""
You are an elite AI Musicologist and Hum-to-Song Recognition Engine.
A user recorded a 10-second vocal hum/singing file. Instead of raw audio noise, we extracted its exact musical profile using Fourier frequency analysis:

- Dominant Musical Pitch Classes (Chroma Notes): {active_notes}
- Melodic Contour Progression (Sequence of relative pitch shifts over time): [{contour_signature}]

Tasks:
1. Cross-reference this exact melodic contour and note progression pattern against world-famous pop, rock, and classic songs (e.g., Beatles tracks like "Let It Be", "Hey Jude", Queen, etc.).
2. Identify the most accurate **Song Title** and **Artist**.
3. Provide a high-precision **Confidence Percentage** (0-100%).
4. Give a clear explanation of how the contour and notes match the melody structure of your prediction.

Respond in a clear, formatted summary.
"""

    response = client.models.generate_content(
        model="gemma-4-31b-it",
        contents=prompt
    )
    
    return response.text

if __name__ == "__main__":
    # 1. Record 10-second audio from microphone
    audio_file = record_audio(duration=10)
    
    # 2. Extract Steps 1, 2, and 3 features and display step-by-step graphs
    contour_sig, notes = process_audio_pipeline(audio_file)
    
    # 3. Predict song using Gemma 4 (31B) (Step 4)
    final_prediction = predict_song_with_gemma(contour_sig, notes)
    
    print("\n==========================================")
    print("🎯 FINAL ANSWER: GEMMA 4 (31B) REASONING REPORT")
    print("==========================================")
    print(final_prediction)