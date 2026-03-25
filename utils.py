import librosa
import numpy as np

def preprocess_audio(file_path, sr=22050):
    """
    Load an audio file (wav, mp3, m4a, mp4), convert to a 128x128 Mel spectrogram,
    and return a batch suitable for CNN input (shape: (1, 128, 128, 1)).
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sr)

        # Generate Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)

        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure fixed size: 128x128
        if mel_spec_db.shape[1] < 128:
            pad_width = 128 - mel_spec_db.shape[1]
            mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_spec_db = mel_spec_db[:, :128]

        # Reshape to (128, 128, 1)
        X = mel_spec_db.reshape(128, 128, 1)

        # Add batch dimension → (1, 128, 128, 1)
        return np.expand_dims(X, axis=0)

    except Exception as e:
        print(f"Error loading audio: {e}")
        return None