# File: evaluate.py (Corrected for KeyError)

import os
import csv
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, accuracy_score

# ===================================================================
# 🔧 CONFIGURATION: YOU MUST FILL THIS IN
# ===================================================================
TEST_DATASET_PATH = "test_dataset/"
LABELS_CSV_PATH = "test_labels.csv"

# ===================================================================
# 🧠 MODEL ARCHITECTURE
# ===================================================================
class EKYCModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=True):
        super(EKYCModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        num_features = self.backbone.num_features
        self.forgery_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
    def forward_one(self, x):
        embedding = self.backbone(x)
        forgery_logit = self.forgery_head(embedding).squeeze(-1)
        return embedding, forgery_logit
    def forward(self, img1, img2):
        embedding1, _ = self.forward_one(img1)
        embedding2, forgery_logit2 = self.forward_one(img2)
        return embedding1, embedding2, forgery_logit2

# ===================================================================
# 🖼️ PREPROCESSING & THRESHOLDS
# ===================================================================
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MATCH_THRESHOLD =0.2
FAKE_THRESHOLD = 0.5

# ===================================================================
#  HELPER FUNCTIONS
# ===================================================================
def load_model(device):
    model = EKYCModel()
    model.load_state_dict(torch.load("model.pt", map_location=device))
    model.to(device)
    model.eval()
    return model

def predict_pair(id_image, selfie_image, model, device):
    id_tensor = transform(id_image).unsqueeze(0).to(device)
    selfie_tensor = transform(selfie_image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb_id, emb_selfie, forgery_logit = model(id_tensor, selfie_tensor)
    is_fake = 1 if torch.sigmoid(forgery_logit).item() > FAKE_THRESHOLD else 0
    distance = torch.nn.functional.pairwise_distance(emb_id, emb_selfie).item()
    is_match = 1 if distance < MATCH_THRESHOLD else 0
    return is_match, is_fake

# ===================================================================
#  MAIN SCRIPT LOGIC
# ===================================================================
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on device: {device}")
    try:
        model = load_model(device)
        print("Model 'model.pt' loaded successfully.")
    except FileNotFoundError:
        print("Error: 'model.pt' not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return

    ground_truth = []
    try:
        # --- THIS IS THE FIX ---
        # Added encoding='utf-8-sig' to handle hidden characters in the CSV header
        with open(LABELS_CSV_PATH, "r", encoding='utf-8-sig') as f:
            reader = csv.DictReader(f)
            ground_truth = list(reader)
        print(f"Loaded {len(ground_truth)} test cases from {LABELS_CSV_PATH}.")
    except FileNotFoundError:
        print(f"Error: Labels CSV not found at '{LABELS_CSV_PATH}'")
        return

    predictions = []
    print("Starting prediction loop...")
    for item in ground_truth:
        try:
            # Strip any whitespace from keys just in case
            item = {k.strip(): v for k, v in item.items()}

            kyc_folder = item['kyc_folder_name']
            selfie_file = item['selfie_filename']
            id_path = os.path.join(TEST_DATASET_PATH, kyc_folder, "id.jpg")
            selfie_path = os.path.join(TEST_DATASET_PATH, kyc_folder, selfie_file)

            id_image = Image.open(id_path).convert("RGB")
            selfie_image = Image.open(selfie_path).convert("RGB")
            pred_match, pred_fake = predict_pair(id_image, selfie_image, model, device)
            predictions.append({
                'true_match': int(item['match_label']),
                'pred_match': pred_match,
                'true_fake': int(item['fake_label']),
                'pred_fake': pred_fake
            })
        except KeyError:
            print(f"  Warning: A row in the CSV has missing or incorrect headers. Skipping. Found headers: {list(item.keys())}")
            continue
        except FileNotFoundError:
            print(f"  Warning: Could not find images for case {kyc_folder}. Skipping.")
            continue
    print("Prediction loop finished.")

    if not predictions:
        print("No predictions were made.")
        return

    y_true_match = [p['true_match'] for p in predictions]
    y_pred_match = [p['pred_match'] for p in predictions]
    y_true_fake = [p['true_fake'] for p in predictions]
    y_pred_fake = [p['pred_fake'] for p in predictions]

    print("\n" + "="*50)
    print("               EVALUATION SUMMARY REPORT")
    print("="*50 + "\n")

    print("--- 👤 Person Match Verification ---")
    print(f"Overall Accuracy: {accuracy_score(y_true_match, y_pred_match):.2%}\n")
    print(classification_report(y_true_match, y_pred_match, target_names=['No Match (0)', 'Match (1)'], zero_division=0))
    
    print("\n--- 🤖 Fake Selfie Detection ---")
    print(f"Overall Accuracy: {accuracy_score(y_true_fake, y_pred_fake):.2%}\n")
    print(classification_report(y_true_fake, y_pred_fake, target_names=['Real (0)', 'Fake (1)'], zero_division=0))
    print("\n" + "="*50)

# ===================================================================
#  EXECUTE
# ===================================================================
if __name__ == "__main__":
    if not TEST_DATASET_PATH or not LABELS_CSV_PATH:
        print("❌ Error: Please edit the script to set the TEST_DATASET_PATH and LABELS_CSV_PATH.")
    else:
        run_evaluation()