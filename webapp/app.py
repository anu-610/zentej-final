from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
import io
import os
import glob

# --- 1. DEFINE YOUR MODEL ARCHITECTURE (CORRECTED) ---
# This MUST match the model used for training model.pt
class EKYCModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b0', pretrained=False):
        super(EKYCModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=0.0)
        num_features = self.backbone.num_features
        self.forgery_head = nn.Sequential(
            nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.5), nn.Linear(256, 1)
        )
    def forward_one(self, x):
        embedding = self.backbone(x)
        forgery_logit = self.forgery_head(embedding).squeeze(-1)
        return embedding, forgery_logit

# --- 2. SETUP AND CONFIGURATION ---
app = Flask(__name__)
device = torch.device("cpu")
DATABASE_FILE = 'user_database.pt'
DUPLICATION_DATA_PATH = "Sentinel_FaceV1/Duplication_Dataset/train/"


# Image transformations
IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Thresholds
FAKE_THRESHOLD = 0.5
MATCH_THRESHOLD = 0.2# This is now the max distance to be considered a match

# --- 3. LOAD MODEL AND USER DATABASE ---

# Load the trained model
model = EKYCModel()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

# Function to create and load the user database
def get_user_database():
    if os.path.exists(DATABASE_FILE):
        print("Loading existing user database...")
        return torch.load(DATABASE_FILE)

    print("Creating new user database... This may take a moment.")
    database = {}
    with torch.no_grad():
        for person_id in os.listdir(DUPLICATION_DATA_PATH):
            person_folder = os.path.join(DUPLICATION_DATA_PATH, person_id)
            if os.path.isdir(person_folder):
                image_files = glob.glob(os.path.join(person_folder, '*.jpg'))
                if not image_files: continue
                
                ref_image_path = image_files[0]
                try:
                    ref_image = Image.open(ref_image_path).convert("RGB")
                    ref_tensor = transform(ref_image).unsqueeze(0).to(device)
                    embedding, _ = model.forward_one(ref_tensor)
                    database[person_id] = embedding.squeeze(0)
                except Exception as e:
                    print(f"Could not process {ref_image_path}: {e}")
    
    torch.save(database, DATABASE_FILE)
    print(f"Database created with {len(database)} users.")
    return database

user_database = get_user_database()
user_ids = list(user_database.keys())
user_embeddings = torch.stack(list(user_database.values()))


# --- 4. DEFINE API ENDPOINTS ---

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    if 'selfie_image' not in request.files:
        return jsonify({'error': 'Missing selfie image file'}), 400

    selfie_image_file = request.files['selfie_image']
    try:
        selfie_image = Image.open(selfie_image_file.stream).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {e}'}), 400

    selfie_tensor = transform(selfie_image).unsqueeze(0).to(device)

    with torch.no_grad():
        selfie_embedding, forgery_logit = model.forward_one(selfie_tensor)
        
        forgery_prob = torch.sigmoid(forgery_logit).item()
        is_fake = forgery_prob > FAKE_THRESHOLD
        
        distances = torch.norm(user_embeddings - selfie_embedding, dim=1)
        min_distance, best_match_index = torch.min(distances, dim=0)
        
        found_user_id = None
        is_match = False
        if min_distance.item() < MATCH_THRESHOLD:
            found_user_id = user_ids[best_match_index]
            is_match = True

    return jsonify({
        'is_match': is_match,
        'user_id': found_user_id,
        'confidence_not_ai': f"{100 * (1 - forgery_prob):.2f}%",
        'authenticity_label': "AI Generated" if is_fake else "Real"
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

