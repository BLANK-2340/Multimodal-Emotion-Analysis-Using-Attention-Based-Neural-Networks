import os
import cv2
import torch
import pandas as pd
import numpy as np
from torchvision import models, transforms
from torch import nn

# Define paths
video_samples_path = r"YOUR_VIDEO_SAMPLES_PATH"
output_excel_path = r"YOUR_OUTPUT_EXCEL_PATH"

# Initialize ResNet-50 model
resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])  # Remove the classification layer
resnet50.eval()

# Define image transformation
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), 512).to(x.device)
        c0 = torch.zeros(1, x.size(0), 512).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Initialize LSTM model
lstm_model = LSTMModel(input_dim=2048, hidden_dim=512, output_dim=512, num_layers=1)
lstm_model.eval()

# Function to extract frames from video
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()
    return frames

# Function to process video and extract feature vector
def process_video(video_path, resnet_model, lstm_model):
    frames = extract_frames(video_path)
    frame_vectors = []
    for frame in frames:
        input_tensor = preprocess(frame).unsqueeze(0)
        with torch.no_grad():
            frame_vector = resnet_model(input_tensor).squeeze().numpy()
        frame_vectors.append(frame_vector)
    frame_tensor = torch.tensor(np.array(frame_vectors)).unsqueeze(0)  # Shape: 1x10x2048
    with torch.no_grad():
        video_vector = lstm_model(frame_tensor).squeeze().numpy()  # Shape: 512
    return video_vector

# Process videos and store features
data = []
for video_file in os.listdir(video_samples_path):
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_samples_path, video_file)
        A2 = process_video(video_path, resnet50, lstm_model)
        
        data.append({
            "File Name": video_file,
            "A2": A2.tolist()
        })

# Convert lists to strings for Excel compatibility
for item in data:
    item["A2"] = str(item["A2"])

# Save to Excel
df = pd.DataFrame(data)
df.to_excel(output_excel_path, index=False)

print("Video feature extraction and saving to Excel completed.")
