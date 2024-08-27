# Required libraries
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import os  # Added for file operations
from itertools import cycle  # Added for ROC curve plotting

# Function to load data from an Excel file
def load_data(excel_path):
    print("Loading data...")
    df = pd.read_excel(excel_path)
    def parse_array(x):
        return np.array(x.strip('[]').split(', '), dtype=np.float32)

    # Parsing and stacking data
    V1 = torch.stack([torch.tensor(parse_array(x)) for x in df['V1']])
    V2 = torch.stack([torch.tensor(parse_array(x)) for x in df['V2']])
    V3 = torch.stack([torch.tensor(parse_array(x)) for x in df['V3']])
    V4 = torch.stack([torch.tensor(parse_array(x)) for x in df['V4']])
    A2 = torch.stack([torch.tensor(parse_array(x)) for x in df['A2']])
    label_mapping = {'anger': 0, 'disgust': 1, 'sadness': 2, 'joy': 3, 'neutral': 4, 'surprise': 5, 'fear': 6}
    Labels = torch.tensor(df['Emotion'].map(label_mapping).values).long()
    print("Data loaded successfully.")
    return V1, V2, V3, V4, A2, Labels, label_mapping


# Emotion analysis model definition
class EmotionAnalysisModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, num_classes):
        super(EmotionAnalysisModel, self).__init__()
        self.alpha1 = nn.Parameter(torch.rand(input_dim))
        self.alpha2 = nn.Parameter(torch.rand(input_dim))
        self.alpha3 = nn.Parameter(torch.rand(input_dim))
        self.bilstm_voice = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.bilstm_text = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.bilstm_video = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.multihead_attention = nn.MultiheadAttention(hidden_dim*2, num_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim*2, num_classes)
        self.project_V3 = nn.Linear(25, input_dim)  # Project V3 from 25 to 512
        self.project_V4 = nn.Linear(25, input_dim)  # Project V4 from 25 to 512

    def forward(self, V1, V2, V3, V4, A2):
        V3_projected = self.project_V3(V3)
        V4_projected = self.project_V4(V4)
        V1_V3_V4 = V1 * self.alpha1 + V3_projected * self.alpha2 + V4_projected * self.alpha3
        F_voice, _ = self.bilstm_voice(V1_V3_V4.unsqueeze(1))
        F_text, _ = self.bilstm_text(V2.unsqueeze(1))
        F_video, _ = self.bilstm_video(A2.unsqueeze(1))
        combined_features = torch.cat((F_voice, F_text, F_video), dim=1)
        attn_output, _ = self.multihead_attention(combined_features, combined_features, combined_features)
        features = self.fc(attn_output[:, -1, :])
        return features

# Training function for the model
def train_model(model, features, labels, epochs=9988):
    print("Starting training...")
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    V1, V2, V3, V4, A2 = features
    model.train()
    train_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = model(V1, V2, V3, V4, A2)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        if epoch % 10 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")
    print("Training completed.")
    return train_losses

# Feature extraction function for use with SVM
def extract_features(model, V1, V2, V3, V4, A2):
    print("Extracting features for SVM...")
    model.eval()  # Ensure the model is in evaluation mode
    with torch.no_grad():
        V3_projected = model.project_V3(V3)
        V4_projected = model.project_V4(V4)
        V1_V3_V4 = V1 * model.alpha1 + V3_projected * model.alpha2 + V4_projected * model.alpha3
        F_voice, _ = model.bilstm_voice(V1_V3_V4.unsqueeze(1))
        F_text, _ = model.bilstm_text(V2.unsqueeze(1))
        F_video, _ = model.bilstm_video(A2.unsqueeze(1))
        combined_features = torch.cat((F_voice, F_text, F_video), dim=1)
        attn_output, _ = model.multihead_attention(combined_features, combined_features, combined_features)
        print("Feature extraction completed.")
        return attn_output[:, -1, :]

# SVM classifier definition
def svm_classifier(features, labels):
    print("Training SVM classifier...")
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    svm = SVC(kernel='rbf', C=1.0, class_weight='balanced', probability=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    y_pred_proba = svm.predict_proba(X_test)
    print("SVM training completed.")
    return X_test, y_test, y_pred, y_pred_proba


# Main execution starts here
excel_path = 'YOUR_EXCEL_PATH'
print("Starting the process...")
V1, V2, V3, V4, A2, Labels, label_mapping = load_data(excel_path)
model = EmotionAnalysisModel(input_dim=512, hidden_dim=256, num_layers=1, num_heads=8, num_classes=7)
print("Model initialized.")

# Training model and timing
start_time = time.time()
train_losses = train_model(model, (V1, V2, V3, V4, A2), Labels)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds.")

# Extract features for SVM
features = extract_features(model, V1, V2, V3, V4, A2).cpu().numpy()
labels = Labels.numpy()

# SVM Classifier and metrics calculation
X_test, y_test, y_pred, y_pred_proba = svm_classifier(features, labels)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print("SVM evaluation metrics calculated.")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix generated.")

# Model size calculation
model_size = sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
print(f"Trainable parameters: {model_size} MB")

# ROC curve and AUC
n_classes = 7
y_test_bin = label_binarize(y_test, classes=[*range(n_classes)])
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
print("ROC curves and AUC calculated.")

# Plotting the ROC curves
print("Plotting ROC curves...")
plt.figure(figsize=(8, 6))
colors = cycle(['blue', 'red', 'green', 'yellow', 'orange', 'purple', 'brown'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))
plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for Multi-Class')
plt.legend(loc="lower right")
plt.show()
print("ROC curves plotted.")

# Plot training loss curve
plt.figure()
plt.plot(train_losses, label='Training loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
print("Training loss curve plotted.")

# Print out results
print(f"Model Summary:\n\n{model}")
print(f"Training took {end_time - start_time:.2f} seconds.")
print(f"Trainable parameters: {model_size:.2f} MB")
print(f"SVM Accuracy: {acc:.4f}")
print(f"SVM Precision: {prec:.4f}")
print(f"SVM Recall: {rec:.4f}")
print(f"SVM F1-Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
print("Confusion matrix plotted.")

# Save model to file
torch.save(model.state_dict(), 'emotion_analysis_model.pth')
print("Model saved to disk as 'emotion_analysis_model.pth'.")

# Calculate and print the model size on disk
model_size_bytes = os.path.getsize('emotion_analysis_model.pth')
model_size_mb = model_size_bytes / (1024 ** 2)
print(f"Model size on disk: {model_size_mb:.2f} MB")


# Calculate and print the number of trainable parameters
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {trainable_params}")

# Calculate and print the model size on disk
torch.save(model.state_dict(), 'emotion_analysis_model.pth')  # Save model to file
model_size_bytes = os.path.getsize('emotion_analysis_model.pth')  # Get file size in bytes
model_size_mb = model_size_bytes / (1024 ** 2)  # Convert bytes to megabytes
print(f"Model size on disk: {model_size_mb:.2f} MB")


def visualize_emotion_classification(y_true, y_pred, label_mapping):

    # Reverse mapping from indices to emotions for readability
    reverse_label_mapping = {v: k for k, v in label_mapping.items()}

    # Compute confusion matrix to get true positives
    cm = confusion_matrix(y_true, y_pred)
    true_positives = cm.diagonal()

    # Prepare data for the DataFrame
    data = []
    for idx, emotion in reverse_label_mapping.items():
        total = sum(cm[idx])
        correct = true_positives[idx]
        wrong = total - correct
        percent_correct = 100 * correct / total if total > 0 else 0
        data.append([emotion, total, correct, wrong, percent_correct])

    # Create and return the DataFrame
    df = pd.DataFrame(data, columns=['Emotion', 'Total', 'Correct', 'Wrong', '% Correct'])
    return df

# Visualizing emotion classification results
emotion_summary = visualize_emotion_classification(y_test, y_pred, label_mapping)
print("Emotion classification results:")
print(emotion_summary)
