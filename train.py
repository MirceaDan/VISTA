import os
import cv2
import time
import math
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

# =========================================================
# CONFIG
# =========================================================
#
# dataset/
#    train/
#       000001.jpg
#       000001.txt
#       ...
#
#    test/
#       000001.jpg
#       000001.txt
#       ...
#
# YOLO LABEL FORMAT:
#
# class x_center y_center width height
#
# normalized [0..1]
#
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 640
BATCH_SIZE = 4
EPOCHS = 100

LR = 1e-4

NUM_CLASSES = 1
MAX_OBJECTS = 64

EPS = 1e-6

GRU_HIDDEN = 32

DATASET_PATH = "./dataset"

TRAIN_PATH = os.path.join(DATASET_PATH, "train")
TEST_PATH  = os.path.join(DATASET_PATH, "test")

MODEL_SAVE_PATH = "./vista_yolo.pth"

# =========================================================
# TRANSFORMS
# =========================================================

train_transform = T.Compose([

    T.Resize((IMG_SIZE, IMG_SIZE)),

    T.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.1
    ),

    T.RandomGrayscale(p=0.15),

    T.RandomHorizontalFlip(p=0.5),

    T.ToTensor()
])

test_transform = T.Compose([

    T.Resize((IMG_SIZE, IMG_SIZE)),

    T.ToTensor()
])

# =========================================================
# UTILS
# =========================================================

def load_yolo_labels(label_path):

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:

        vals = line.strip().split()

        if len(vals) != 5:
            continue

        cls, x, y, w, h = map(float, vals)

        boxes.append([
            int(cls),
            x,
            y,
            w,
            h
        ])

    return boxes


def compute_motion_features(curr_box, prev_box):

    x, y, w, h = curr_box
    px, py, pw, ph = prev_box

    area = w * h
    prev_area = pw * ph

    delta_x = x - px
    delta_y = y - py

    growth = math.log(
        (area + EPS) /
        (prev_area + EPS)
    )

    return np.array([
        x,
        y,
        w,
        h,
        delta_x,
        delta_y,
        growth
    ], dtype=np.float32)

# =========================================================
# DATASET
# =========================================================

class VistaYOLODataset(Dataset):

    def __init__(self, dataset_path, transform=None):

        self.dataset_path = dataset_path
        self.transform = transform

        self.images = []

        for f in os.listdir(dataset_path):

            if f.endswith(".jpg") or f.endswith(".png"):

                self.images.append(f)

        self.images.sort()

    def __len__(self):

        return len(self.images) - 1

    def load_image(self, idx):

        img_path = os.path.join(
            self.dataset_path,
            self.images[idx]
        )

        img = cv2.imread(img_path)

        if img is None:
            raise RuntimeError(f"Cannot load {img_path}")

        img_rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        pil = Image.fromarray(img_rgb)

        return img, self.transform(pil)

    def load_labels(self, idx):

        img_name = self.images[idx]

        label_name = (
            img_name
            .replace(".jpg", ".txt")
            .replace(".png", ".txt")
        )

        label_path = os.path.join(
            self.dataset_path,
            label_name
        )

        return load_yolo_labels(label_path)

    def __getitem__(self, idx):

        frame_t_raw, frame_t = self.load_image(idx)

        frame_prev_raw, frame_prev = self.load_image(
            max(idx - 1, 0)
        )

        labels_t = self.load_labels(idx)

        labels_prev = self.load_labels(
            max(idx - 1, 0)
        )

        motion_features = []

        targets = []

        num_boxes = min(
            len(labels_t),
            len(labels_prev),
            MAX_OBJECTS
        )

        for i in range(num_boxes):

            cls, x, y, w, h = labels_t[i]

            _, px, py, pw, ph = labels_prev[i]

            mf = compute_motion_features(
                [x, y, w, h],
                [px, py, pw, ph]
            )

            motion_features.append(mf)

            targets.append([
                cls,
                x,
                y,
                w,
                h
            ])

        if len(motion_features) == 0:

            motion_features = np.zeros(
                (1,7),
                dtype=np.float32
            )

            targets = np.zeros(
                (1,5),
                dtype=np.float32
            )

        motion_features = torch.tensor(
            motion_features,
            dtype=torch.float32
        )

        targets = torch.tensor(
            targets,
            dtype=torch.float32
        )

        return (
            frame_t,
            frame_prev,
            motion_features,
            targets
        )

# =========================================================
# COLLATE
# =========================================================

def collate_fn(batch):

    imgs = []
    prevs = []
    motions = []
    targets = []

    for img, prev, motion, target in batch:

        imgs.append(img)
        prevs.append(prev)

        motions.append(motion)
        targets.append(target)

    return (
        torch.stack(imgs),
        torch.stack(prevs),
        motions,
        targets
    )

# =========================================================
# BACKBONE
# =========================================================

class ConvBlock(nn.Module):

    def __init__(self, c1, c2, k=3, s=1):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                c1,
                c2,
                k,
                s,
                k//2,
                bias=False
            ),

            nn.BatchNorm2d(c2),

            nn.SiLU()
        )

    def forward(self, x):

        return self.block(x)


class YOLOBackbone(nn.Module):

    def __init__(self):

        super().__init__()

        self.net = nn.Sequential(

            ConvBlock(3, 32, 3, 2),

            ConvBlock(32, 64, 3, 2),

            ConvBlock(64, 128, 3, 2),

            ConvBlock(128, 128),

            ConvBlock(128, 256, 3, 2),

            ConvBlock(256, 256),

            ConvBlock(256, 512, 3, 2),

            ConvBlock(512, 512)
        )

    def forward(self, x):

        return self.net(x)

# =========================================================
# MOTION GRU
# =========================================================

class MotionGRU(nn.Module):

    def __init__(
        self,
        input_size=7,
        hidden_size=GRU_HIDDEN
    ):

        super().__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

    def forward(self, x):

        out, h = self.gru(x)

        return h[-1]

# =========================================================
# STATICNESS HEAD
# =========================================================

class StaticnessHead(nn.Module):

    def __init__(self, in_features):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(in_features, 64),

            nn.SiLU(),

            nn.Linear(64, 32),

            nn.SiLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):

        return torch.sigmoid(
            self.net(x)
        )

# =========================================================
# DETECTION HEAD
# =========================================================

class DetectionHead(nn.Module):

    def __init__(self, c=512):

        super().__init__()

        self.head = nn.Sequential(

            ConvBlock(c, 256),

            nn.Conv2d(
                256,
                NUM_CLASSES + 5,
                1
            )
        )

    def forward(self, x):

        return self.head(x)

# =========================================================
# VISTA YOLO
# =========================================================

class VISTA_YOLO(nn.Module):

    def __init__(self):

        super().__init__()

        self.backbone = YOLOBackbone()

        self.motion_encoder = MotionGRU()

        self.staticness = StaticnessHead(
            GRU_HIDDEN
        )

        self.det_head = DetectionHead()

    def forward(self, x, motion_features):

        feat = self.backbone(x)

        B, C, H, W = feat.shape

        staticness_scores = []

        for b in range(B):

            mf = motion_features[b]

            if len(mf.shape) == 2:

                mf = mf.unsqueeze(0)

            motion_embedding = self.motion_encoder(
                mf
            )

            staticness = self.staticness(
                motion_embedding
            )

            staticness_scores.append(
                staticness
            )

        staticness_scores = torch.stack(
            staticness_scores
        )

        det = self.det_head(feat)

        staticness_map = staticness_scores.view(
            B,
            1,
            1,
            1
        )

        # objectness modulation
        det[:,4:5,:,:] = (
            det[:,4:5,:,:] *
            staticness_map
        )

        return det, staticness_scores

# =========================================================
# LOSS
# =========================================================

class VistaLoss(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(
        self,
        pred,
        staticness,
        targets
    ):

        # placeholder detection loss
        det_loss = torch.mean(pred**2)

        static_loss = 0.0

        for b in range(len(targets)):

            if len(targets[b]) > 0:

                gt_static = torch.ones_like(
                    staticness[b]
                )

                static_loss += F.binary_cross_entropy(
                    staticness[b],
                    gt_static
                )

        total = det_loss + 0.5 * static_loss

        return total

# =========================================================
# DATASETS
# =========================================================

train_dataset = VistaYOLODataset(
    TRAIN_PATH,
    train_transform
)

test_dataset = VistaYOLODataset(
    TEST_PATH,
    test_transform
)

# =========================================================
# DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# =========================================================
# MODEL
# =========================================================

model = VISTA_YOLO().to(DEVICE)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

criterion = VistaLoss()

# =========================================================
# TRAIN LOOP
# =========================================================

def train_one_epoch():

    model.train()

    total_loss = 0.0

    for img, prev, motion, target in train_loader:

        img = img.to(DEVICE)

        motion = [
            m.to(DEVICE)
            for m in motion
        ]

        pred, staticness = model(
            img,
            motion
        )

        loss = criterion(
            pred,
            staticness,
            target
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================================================
# TEST LOOP
# =========================================================

def evaluate():

    model.eval()

    total_loss = 0.0

    staticness_mean = 0.0

    with torch.no_grad():

        for img, prev, motion, target in test_loader:

            img = img.to(DEVICE)

            motion = [
                m.to(DEVICE)
                for m in motion
            ]

            pred, staticness = model(
                img,
                motion
            )

            loss = criterion(
                pred,
                staticness,
                target
            )

            total_loss += loss.item()

            staticness_mean += (
                staticness.mean().item()
            )

    total_loss /= len(test_loader)

    staticness_mean /= len(test_loader)

    return total_loss, staticness_mean

# =========================================================
# MAIN TRAINING
# =========================================================

print("=== TRAINING VISTA YOLO ===")

best_test_loss = 999999

for epoch in range(EPOCHS):

    start = time.time()

    train_loss = train_one_epoch()

    test_loss, staticness_score = evaluate()

    elapsed = (
        time.time() - start
    ) / 60.0

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Test Loss: {test_loss:.4f} | "
        f"Staticness: {staticness_score:.4f} | "
        f"{elapsed:.2f} min"
    )

    # =====================================================
    # SAVE BEST MODEL
    # =====================================================

    if test_loss < best_test_loss:

        best_test_loss = test_loss

        model.eval()

        scripted = torch.jit.script(model)

        scripted.save(MODEL_SAVE_PATH)

        print(
            f"Best model saved: "
            f"{MODEL_SAVE_PATH}"
        )

print("=== TRAINING FINISHED ===")