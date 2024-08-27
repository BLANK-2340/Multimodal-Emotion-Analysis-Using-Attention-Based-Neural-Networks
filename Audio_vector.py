import os
import pandas as pd
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from torchaudio.transforms import Resample, MFCC
import librosa

# Define paths
video_samples_path = r"YOUR_VIDEO_SAMPLES_PATH"
output_excel_path = r"YOUR_OUTPUT_EXCEL_PATH"

# Initialize Wav2Vec2 processor and model
voice_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
voice_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").eval()

# Function to extract audio from video
def extract_audio_from_video(video_path):
    audio_path = video_path.replace('.mp4', '.wav')
    command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_path}\" -y"
    os.system(command)
    return audio_path

# V1 vector (voice embedding) 1x512
def extract_voice_embeddings(audio_path, processor, model):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16000:
        resampler = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    input_values = processor(waveform.squeeze(), return_tensors="pt", sampling_rate=16000).input_values
    with torch.no_grad():
        hidden_states = model(input_values).last_hidden_state
    embeddings = hidden_states.mean(dim=1)
    return embeddings.squeeze().numpy()[:512]

# V3 vector (MFCCS) 1x25
def extract_mfccs(audio_path, n_mfcc=25):
    waveform, sample_rate = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    mfcc_transform = MFCC(sample_rate=sample_rate, n_mfcc=n_mfcc)
    mfcc = mfcc_transform(waveform).mean(dim=2).squeeze()
    return mfcc.numpy()[:25]

# V4 vector (spectral centroid, zero cross rate, pitch, RMS energy, and tempo) 1x25
def extract_combined_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y).mean()
    rms = librosa.feature.rms(y=y).mean()
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch = np.mean(pitches[pitches > 0])
    feature_vector = np.array([spectral_centroid, zero_crossing_rate, rms, tempo, pitch])
    feature_matrix = np.tile(feature_vector, (5, 1))
    flattened_features = feature_matrix.flatten()
    return flattened_features[:25]

# Process videos and store features
data = []
for video_file in os.listdir(video_samples_path):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_samples_path, video_file)
        audio_path = extract_audio_from_video(video_path)
        
        V1 = extract_voice_embeddings(audio_path, voice_processor, voice_model)
        V3 = extract_mfccs(audio_path)
        V4 = extract_combined_features(audio_path)
        
        data.append({
            "File Name": video_file,
            "V1": V1.tolist(),
            "V3": V3.tolist(),
            "V4": V4.tolist()
        })

# Convert lists to strings for Excel compatibility
for item in data:
    item["V1"] = str(item["V1"])
    item["V3"] = str(item["V3"])
    item["V4"] = str(item["V4"])

# Save to Excel
df = pd.DataFrame(data)
df.to_excel(output_excel_path, index=False)

print("Audio feature extraction and saving to Excel completed.")
