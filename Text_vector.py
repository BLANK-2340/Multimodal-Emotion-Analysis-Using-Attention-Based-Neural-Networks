import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

# Define paths 
utterances_df_path = r"YOUR_UTTERANCES_DF_PATH"
output_excel_path = r"YOUR_OUTPUT_EXCEL_PATH"

# Load utterances DataFrame
utterances_df = pd.read_excel(utterances_df_path, usecols=['Filename', 'Utterance'])
utterances_df['Filename'] = utterances_df['Filename'].apply(lambda x: f"{x}.mp4")

# Ensure utterances are strings
utterances_df['Utterance'] = utterances_df['Utterance'].astype(str)

# Initialize text model
text_model = SentenceTransformer('all-MiniLM-L6-v2').eval()

# V2 vector (utterance) 1x512
def extract_utterance_embeddings(text, model):
    with torch.no_grad():
        embeddings = model.encode([text], convert_to_tensor=True, show_progress_bar=False)
    return embeddings.squeeze().numpy()

# Process text and store features
data = []
for index, row in utterances_df.iterrows():
    video_file = row['Filename']
    utterance_text = row['Utterance']
    V2 = extract_utterance_embeddings(utterance_text, text_model)
    
    data.append({
        "File Name": video_file,
        "V2": V2.tolist()
    })

# Convert lists to strings for Excel compatibility
for item in data:
    item["V2"] = str(item["V2"])

# Save to Excel
df = pd.DataFrame(data)
df.to_excel(output_excel_path, index=False)

print("Text feature extraction and saving to Excel completed.")

# to get V2 vector in dimension 1x512
