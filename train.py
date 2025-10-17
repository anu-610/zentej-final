# File: train.py (Corrected Version)

import torch
import torch.nn as nn
import timm
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import PairedFaceDataset

# --- MODEL, LOSS, AND PREPROCESSING DEFINITIONS (No changes here) ---
class EKYCModel(nn.Module):
    def __init__(self, backbone_name='efficientnet_b2', pretrained=True):
        super(EKYCModel, self).__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, drop_rate=0.3)
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

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    def forward(self, emb1, emb2, label):
        euclidean_distance = torch.nn.functional.pairwise_distance(emb1, emb2)
        loss = (1 - label) * torch.pow(euclidean_distance, 2) + \
               (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        return torch.mean(loss)

IMG_SIZE = 224
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# --- WRAP THE TRAINING LOGIC IN A FUNCTION ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparameters
    NUM_EPOCHS = 20
    BATCH_SIZE = 32 # Using the safer batch size
    LEARNING_RATE = 0.0001
    LOSS_ALPHA = 0.6

    train_dataset = PairedFaceDataset(csv_file='train_labels.csv', transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)

    model = EKYCModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    identity_loss_fn = ContrastiveLoss()
    forgery_loss_fn = nn.BCEWithLogitsLoss()

    scaler = torch.cuda.amp.GradScaler()
    print("Starting training with Automatic Mixed Precision...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for i, batch in enumerate(train_loader):
            img1 = batch['image1'].to(device)
            img2 = batch['image2'].to(device)
            identity_label = batch['identity_label'].to(device)
            forgery_label = batch['forgery_label'].to(device)
            
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                emb1, emb2, forgery_logit = model(img1, img2)
                identity_loss = identity_loss_fn(emb1, emb2, identity_label)
                forgery_loss = forgery_loss_fn(forgery_logit, forgery_label)
                total_loss = (LOSS_ALPHA * identity_loss) + ((1 - LOSS_ALPHA) * forgery_loss)

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += total_loss.item()
            
            if (i + 1) % 50 == 0:
                 print(f"  Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(train_loader)}], Loss: {total_loss.item():.4f}")

        print(f"--- End of Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader):.4f} ---")

    torch.save(model.state_dict(), "model.pt")
    print("\nTraining complete! Model saved as model.pt")


# --- THIS IS THE CRITICAL FIX ---
# Only run the training function if this script is executed directly
if __name__ == "__main__":
    train_model()