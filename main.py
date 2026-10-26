
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt


def process_audio_pipeline(audio_path, sr=22050):
    """
    Process an uploaded audio file and generate:
    1. Waveform
    2. Spectrogram
    3. Chroma features

    No microphone recording or Gemma/API calls.
    """

    # Load the audio file
    y, sr = librosa.load(audio_path, sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError("The uploaded audio file is empty.")

    # 1. Waveform
    waveform_fig, ax = plt.subplots(figsize=(10, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Audio Waveform")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Amplitude")
    waveform_fig.tight_layout()

    # 2. Spectrogram
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    spectrogram_fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        S_db,
        sr=sr,
        x_axis="time",
        y_axis="log",
        ax=ax
    )
    ax.set_title("Audio Spectrogram")
    spectrogram_fig.colorbar(
        ax.collections[0],
        ax=ax,
        format="%+2.0f dB"
    )
    spectrogram_fig.tight_layout()

    # 3. Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

    chroma_fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.specshow(
        chroma,
        sr=sr,
        x_axis="time",
        y_axis="chroma",
        ax=ax
    )
    ax.set_title("Chroma Features")
    chroma_fig.colorbar(ax.collections[0], ax=ax)
    chroma_fig.tight_layout()

    # Extract the strongest pitch classes as audio features
    pitch_names = [
        "C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"
    ]

    average_chroma = np.mean(chroma, axis=1)
    strongest_indices = np.argsort(average_chroma)[::-1][:4]

    active_notes = [
        pitch_names[int(index)]
        for index in strongest_indices
    ]

    return {
        "duration": librosa.get_duration(y=y, sr=sr),
        "sample_rate": sr,
        "waveform": waveform_fig,
        "spectrogram": spectrogram_fig,
        "chroma": chroma_fig,
        "active_notes": active_notes,
        "chroma_values": average_chroma.tolist()
    }
