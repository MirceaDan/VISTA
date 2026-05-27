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
import torchvision.models as models

# =========================================================
# CONFIG
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
TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train"
)
TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
)
MODEL_SAVE_PATH = "./vista_resnet_flow_ssd.pth"

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
    dx = x - px
    dy = y - py
    growth = math.log(
        (area + EPS) /
        (prev_area + EPS)
    )
    return np.array([
        x,
        y,
        w,
        h,
        dx,
        dy,
        growth
    ], dtype=np.float32)

# =========================================================
# DATASET
# =========================================================
class VistaSSDDataset(Dataset):
    def __init__(
        self,
        dataset_path,
        transform=None
    ):
        self.dataset_path = dataset_path
        self.transform = transform
        self.images = []
        for f in os.listdir(dataset_path):
            if (
                f.endswith(".jpg") or
                f.endswith(".png")
            ):
                self.images.append(f)
        self.images.sort()

    def __len__(self):
        return len(self.images) - 1

    def load_image(self, idx):
        path = os.path.join(
            self.dataset_path,
            self.images[idx]
        )
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(path)

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
        path = os.path.join(
            self.dataset_path,
            label_name
        )
        return load_yolo_labels(path)

    def compute_optical_flow(
        self,
        curr,
        prev
    ):
        curr_gray = cv2.cvtColor(
            curr,
            cv2.COLOR_BGR2GRAY
        )
        prev_gray = cv2.cvtColor(
            prev,
            cv2.COLOR_BGR2GRAY
        )
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        flow = cv2.resize(
            flow,
            (IMG_SIZE, IMG_SIZE)
        )
        flow = torch.tensor(
            flow,
            dtype=torch.float32
        )
        flow = flow.permute(2,0,1)
        return flow

    def __getitem__(self, idx):
        prev_raw, prev_img = self.load_image(
            max(idx - 1, 0)
        )
        curr_raw, curr_img = self.load_image(
            idx
        )
        flow = self.compute_optical_flow(
            curr_raw,
            prev_raw
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
            curr_img,
            prev_img,
            flow,
            motion_features,
            targets
        )

# =========================================================
# COLLATE
# =========================================================
def collate_fn(batch):
    imgs = []
    prevs = []
    flows = []
    motions = []
    targets = []
    for img, prev, flow, motion, target in batch:
        imgs.append(img)
        prevs.append(prev)
        flows.append(flow)
        motions.append(motion)
        targets.append(target)
    return (
        torch.stack(imgs),
        torch.stack(prevs),
        torch.stack(flows),
        motions,
        targets
    )

# =========================================================
# RESNET18 BACKBONE
# =========================================================
class ResNet18Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
        )
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool
        )
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5

# =========================================================
# FLOW ENCODER
# =========================================================
class FlowEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# MOTION FUSION
# =========================================================
class MotionFusion(nn.Module):
    def __init__(
        self,
        rgb_channels,
        flow_channels=128
    ):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv2d(
                flow_channels,
                rgb_channels,
                1
            ),
            nn.Sigmoid()
        )

    def forward(
        self,
        rgb_feat,
        flow_feat
    ):
        A = self.attn(flow_feat)
        return rgb_feat * (1.0 + A)

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
# SSD HEAD
# =========================================================
class SSDHead(nn.Module):
    def __init__(
        self,
        in_channels,
        num_anchors=6
    ):
        super().__init__()
        self.loc_head = nn.Conv2d(
            in_channels,
            num_anchors * 4,
            3,
            padding=1
        )
        self.cls_head = nn.Conv2d(
            in_channels,
            num_anchors * (NUM_CLASSES + 1),
            3,
            padding=1
        )
        self.obj_head = nn.Conv2d(
            in_channels,
            num_anchors,
            3,
            padding=1
        )

    def forward(self, x):
        loc = self.loc_head(x)
        cls = self.cls_head(x)
        obj = self.obj_head(x)
        return loc, cls, obj

# =========================================================
# VISTA SSD
# =========================================================
class VISTA_RESNET_FLOW_SSD(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.flow_encoder = FlowEncoder()
        self.fusion_c3 = MotionFusion(128)
        self.fusion_c4 = MotionFusion(256)
        self.fusion_c5 = MotionFusion(512)
        self.motion_encoder = MotionGRU()
        self.staticness = StaticnessHead(
            GRU_HIDDEN
        )
        self.ssd_c3 = SSDHead(128)
        self.ssd_c4 = SSDHead(256)
        self.ssd_c5 = SSDHead(512)

    def forward(
        self,
        x,
        flow,
        motion_features
    ):
        c3, c4, c5 = self.backbone(x)
        flow_feat = self.flow_encoder(flow)
        flow_c3 = F.interpolate(
            flow_feat,
            size=c3.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        flow_c4 = F.interpolate(
            flow_feat,
            size=c4.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        flow_c5 = F.interpolate(
            flow_feat,
            size=c5.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        c3 = self.fusion_c3(c3, flow_c3)
        c4 = self.fusion_c4(c4, flow_c4)
        c5 = self.fusion_c5(c5, flow_c5)
        B = x.shape[0]
        staticness_scores = []
        for b in range(B):
            mf = motion_features[b]
            if len(mf.shape) == 2:
                mf = mf.unsqueeze(0)
            emb = self.motion_encoder(mf)
            s = self.staticness(emb)
            staticness_scores.append(s)
        staticness_scores = torch.stack(
            staticness_scores
        )
        loc3, cls3, obj3 = self.ssd_c3(c3)
        loc4, cls4, obj4 = self.ssd_c4(c4)
        loc5, cls5, obj5 = self.ssd_c5(c5)
        gate = staticness_scores.view(
            B,
            1,
            1,
            1
        )
        obj3 *= gate
        obj4 *= gate
        obj5 *= gate
        return {
            "loc": [loc3, loc4, loc5],
            "cls": [cls3, cls4, cls5],
            "obj": [obj3, obj4, obj5],
            "staticness": staticness_scores
        }

# =========================================================
# LOSS
# =========================================================
class VistaSSDLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        outputs,
        targets
    ):
        loc_loss = 0.0
        cls_loss = 0.0
        obj_loss = 0.0
        for loc in outputs["loc"]:
            loc_loss += torch.mean(
                loc ** 2
            )

        for cls in outputs["cls"]:
            cls_loss += torch.mean(
                cls ** 2
            )

        for obj in outputs["obj"]:
            obj_loss += torch.mean(
                obj ** 2
            )

        static_loss = 0.0
        staticness = outputs["staticness"]
        for b in range(len(targets)):
            gt_static = torch.ones_like(
                staticness[b]
            )
            static_loss += F.binary_cross_entropy(
                staticness[b],
                gt_static
            )

        total = (
            1.0 * loc_loss +
            1.0 * cls_loss +
            0.5 * obj_loss +
            0.5 * static_loss
        )

        return total

# =========================================================
# DATASETS
# =========================================================
train_dataset = VistaSSDDataset(
    TRAIN_PATH,
    train_transform
)

test_dataset = VistaSSDDataset(
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
model = VISTA_RESNET_FLOW_SSD().to(DEVICE)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)
criterion = VistaSSDLoss()

# =========================================================
# TRAIN
# =========================================================
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for img, prev, flow, motion, target in train_loader:
        img = img.to(DEVICE)
        flow = flow.to(DEVICE)
        motion = [
            m.to(DEVICE)
            for m in motion
        ]
        outputs = model(
            img,
            flow,
            motion
        )
        loss = criterion(
            outputs,
            target
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# =========================================================
# EVALUATION
# =========================================================
def evaluate():
    model.eval()
    total_loss = 0.0
    staticness_mean = 0.0
    with torch.no_grad():
        for img, prev, flow, motion, target in test_loader:
            img = img.to(DEVICE)
            flow = flow.to(DEVICE)
            motion = [
                m.to(DEVICE)
                for m in motion
            ]
            outputs = model(
                img,
                flow,
                motion
            )
            loss = criterion(
                outputs,
                target
            )
            total_loss += loss.item()
            staticness_mean += (
                outputs["staticness"]
                .mean()
                .item()
            )
    total_loss /= len(test_loader)
    staticness_mean /= len(test_loader)
    return total_loss, staticness_mean

# =========================================================
# MAIN TRAINING
# =========================================================
print("=== TRAINING VISTA RESNET FLOW SSD ===")
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
    if test_loss < best_test_loss:
        best_test_loss = test_loss
        model.eval()
        scripted = torch.jit.script(model)
        scripted.save(
            MODEL_SAVE_PATH
        )
        print(
            f"Best model saved: "
            f"{MODEL_SAVE_PATH}"
        )
print("=== TRAINING FINISHED ===")