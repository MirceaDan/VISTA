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
IMG_SIZE = 512
BATCH_SIZE = 1
EPOCHS = 30
LR = 1e-5
NUM_CLASSES = 1
NUM_QUERIES = 25
MAX_OBJECTS = 64
EPS = 1e-6
GRU_HIDDEN = 32
D_MODEL = 256
NHEADS = 8
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
FLOW_ALPHA = 0.35
STATICNESS_LAMBDA = 5.0
DATASET_PATH = "./dataset"
TRAIN_PATH = os.path.join(
    DATASET_PATH,
    "train"
)
TEST_PATH = os.path.join(
    DATASET_PATH,
    "test"
)
MODEL_SAVE_PATH = "./vistadetr_best.pth"

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
    T.RandomHorizontalFlip(p=0.5),
    T.RandomGrayscale(p=0.15),
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
    velocity = math.sqrt(
        dx * dx +
        dy * dy
    )
    return np.array([
        x,
        y,
        w,
        h,
        dx,
        dy,
        growth
    ], dtype=np.float32), velocity

# =========================================================
# DATASET
# =========================================================
class VistaDETRDataset(Dataset):
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
        rgb = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
        pil = Image.fromarray(rgb)
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
        velocities = []
        num_boxes = min(
            len(labels_t),
            len(labels_prev),
            MAX_OBJECTS
        )
        for i in range(num_boxes):
            cls, x, y, w, h = labels_t[i]
            _, px, py, pw, ph = labels_prev[i]
            mf, velocity = compute_motion_features(
                [x, y, w, h],
                [px, py, pw, ph]
            )
            motion_features.append(mf)
            velocities.append(velocity)
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
            velocities = [0.0]

        staticness_gt = []
        for v in velocities:
            s = math.exp(
                -STATICNESS_LAMBDA * v
            )
            staticness_gt.append([s])

        motion_features = torch.tensor(
            motion_features,
            dtype=torch.float32
        )
        targets = torch.tensor(
            targets,
            dtype=torch.float32
        )
        staticness_gt = torch.tensor(
            staticness_gt,
            dtype=torch.float32
        )
        return (
            curr_img,
            prev_img,
            flow,
            motion_features,
            targets,
            staticness_gt
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
    staticness = []
    for img, prev, flow, motion, target, s in batch:
        imgs.append(img)
        prevs.append(prev)
        flows.append(flow)
        motions.append(motion)
        targets.append(target)
        staticness.append(s)

    return (
        torch.stack(imgs),
        torch.stack(prevs),
        torch.stack(flows),
        motions,
        targets,
        staticness
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
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

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
            nn.SiLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)

# =========================================================
# FLOW ATTENTION FUSION
# =========================================================
class FlowAttentionFusion(nn.Module):
    def __init__(
        self,
        feat_channels=512,
        flow_channels=256
    ):
        super().__init__()
        self.flow_proj = nn.Conv2d(
            flow_channels,
            feat_channels,
            1
        )
        self.attn = nn.Sequential(
            nn.Conv2d(
                feat_channels,
                feat_channels,
                1
            ),
            nn.Sigmoid()
        )

    def forward(
        self,
        feat,
        flow
    ):
        flow = F.interpolate(
            flow,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        flow = self.flow_proj(flow)
        A = self.attn(flow)
        feat = feat + FLOW_ALPHA * feat * A
        return feat

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
# POSITION ENCODING
# =========================================================
class PositionEmbedding(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.row_embed = nn.Parameter(
            torch.rand(128, d_model // 2)
        )
        self.col_embed = nn.Parameter(
            torch.rand(128, d_model // 2)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        rows = self.row_embed[:h]
        cols = self.col_embed[:w]
        pos = torch.cat([
            cols.unsqueeze(0).repeat(h,1,1),
            rows.unsqueeze(1).repeat(1,w,1)
        ], dim=-1)
        pos = pos.permute(2,0,1)
        return pos.unsqueeze(0)

# =========================================================
# DETR HEAD
# =========================================================
class DETRHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Conv2d(
            512,
            D_MODEL,
            1
        )
        self.flow_bias_proj = nn.Conv2d(
            256,
            D_MODEL,
            1
        )
        self.position = PositionEmbedding(
            D_MODEL
        )
        self.query_embed = nn.Embedding(
            NUM_QUERIES,
            D_MODEL
        )
        self.transformer = nn.Transformer(
            d_model=D_MODEL,
            nhead=NHEADS,
            num_encoder_layers=NUM_ENCODER_LAYERS,
            num_decoder_layers=NUM_DECODER_LAYERS
        )
        self.class_embed = nn.Linear(
            D_MODEL,
            NUM_CLASSES + 1
        )
        self.bbox_embed = nn.Sequential(
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, D_MODEL),
            nn.ReLU(),
            nn.Linear(D_MODEL, 4)
        )

    def forward(
        self,
        feat,
        flow_feat
    ):
        B = feat.shape[0]
        feat = self.input_proj(feat)
        flow_bias = self.flow_bias_proj(
            flow_feat
        )
        flow_bias = F.interpolate(
            flow_bias,
            size=feat.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        pos = self.position(feat).to(
            feat.device
        )
        src = feat + pos + flow_bias
        src = src.flatten(2).permute(2,0,1)
        query = self.query_embed.weight.unsqueeze(1)
        query = query.repeat(1,B,1)
        tgt = torch.zeros_like(query)
        hs = self.transformer(
            src=src,
            tgt=tgt + query
        )
        hs = hs.permute(1,0,2)
        pred_logits = self.class_embed(hs)
        pred_boxes = torch.sigmoid(
            self.bbox_embed(hs)
        )

        return pred_logits, pred_boxes

# =========================================================
# VISTA DETR
# =========================================================
class VISTA_DETR(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ResNet18Backbone()
        self.backbone.requires_grad_(False)
        self.flow_encoder = FlowEncoder()
        self.flow_fusion = FlowAttentionFusion()
        self.motion_encoder = MotionGRU()
        self.staticness = StaticnessHead(
            GRU_HIDDEN
        )
        self.detr = DETRHead()

    def forward(
        self,
        img,
        flow,
        motion_features
    ):
        feat = self.backbone(img)
        flow_feat = self.flow_encoder(flow)
        feat = self.flow_fusion(
            feat,
            flow_feat
        )
        B = img.shape[0]
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
        pred_logits, pred_boxes = self.detr(
            feat,
            flow_feat
        )
        gate = staticness_scores.view(
            B,
            1,
            1
        )
        pred_logits = pred_logits * gate
        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "staticness": staticness_scores
        }

# =========================================================
# GIOU
# =========================================================
def box_cxcywh_to_xyxy(x):
    cx, cy, w, h = x.unbind(-1)
    b = [
        cx - 0.5 * w,
        cy - 0.5 * h,
        cx + 0.5 * w,
        cy + 0.5 * h
    ]

    return torch.stack(b, dim=-1)


def generalized_iou(boxes1, boxes2):
    x1 = torch.max(boxes1[:,0], boxes2[:,0])
    y1 = torch.max(boxes1[:,1], boxes2[:,1])
    x2 = torch.min(boxes1[:,2], boxes2[:,2])
    y2 = torch.min(boxes1[:,3], boxes2[:,3])
    inter = (x2 - x1).clamp(min=0) * (
        y2 - y1
    ).clamp(min=0)
    area1 = (
        (boxes1[:,2] - boxes1[:,0]) *
        (boxes1[:,3] - boxes1[:,1])
    )
    area2 = (
        (boxes2[:,2] - boxes2[:,0]) *
        (boxes2[:,3] - boxes2[:,1])
    )
    union = area1 + area2 - inter
    iou = inter / (union + EPS)
    cx1 = torch.min(boxes1[:,0], boxes2[:,0])
    cy1 = torch.min(boxes1[:,1], boxes2[:,1])
    cx2 = torch.max(boxes1[:,2], boxes2[:,2])
    cy2 = torch.max(boxes1[:,3], boxes2[:,3])
    c_area = (
        (cx2 - cx1) *
        (cy2 - cy1)
    )
    giou = iou - (
        (c_area - union) /
        (c_area + EPS)
    )

    return giou

# =========================================================
# LOSS
# =========================================================
class VistaDETRLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs,
        targets,
        staticness_gt
    ):
        pred_logits = outputs["pred_logits"]
        pred_boxes = outputs["pred_boxes"]
        pred_staticness = outputs["staticness"]
        B = pred_logits.shape[0]
        total_cls = 0.0
        total_bbox = 0.0
        total_giou = 0.0
        total_static = 0.0
        for b in range(B):
            tgt = targets[b]
            tgt_boxes = tgt[:,1:5].to(
                pred_boxes.device
            )
            num_tgt = min(
                tgt_boxes.shape[0],
                NUM_QUERIES
            )
            if num_tgt == 0:
                continue
            cls_target = torch.zeros(
                NUM_QUERIES,
                dtype=torch.long,
                device=pred_logits.device
            )
            cls_target[:num_tgt] = 1
            total_cls += self.cls_loss(
                pred_logits[b],
                cls_target
            )
            pred_b = pred_boxes[b,:num_tgt]
            total_bbox += F.l1_loss(
                pred_b,
                tgt_boxes[:num_tgt]
            )
            pxy = box_cxcywh_to_xyxy(pred_b)
            gxy = box_cxcywh_to_xyxy(
                tgt_boxes[:num_tgt]
            )
            giou = generalized_iou(
                pxy,
                gxy
            )
            total_giou += (
                1.0 - giou.mean()
            )
            gt_static = staticness_gt[b].mean().to(
                pred_staticness.device
            )
            total_static += F.mse_loss(
                pred_staticness[b].mean(),
                gt_static
            )
        total = (
            1.0 * total_cls +
            5.0 * total_bbox +
            2.0 * total_giou +
            0.5 * total_static
        )

        return total

# =========================================================
# DATASETS
# =========================================================
train_dataset = VistaDETRDataset(
    TRAIN_PATH,
    train_transform
)

test_dataset = VistaDETRDataset(
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
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True
)

# =========================================================
# MODEL
# =========================================================
model = VISTA_DETR().to(DEVICE)
optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=LR,
    weight_decay=1e-4
)

criterion = VistaDETRLoss()

# =========================================================
# TRAIN
# =========================================================
def train_one_epoch():
    model.train()
    total_loss = 0.0
    for img, prev, flow, motion, target, static_gt in train_loader:
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
            target,
            static_gt
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
        for img, prev, flow, motion, target, static_gt in test_loader:
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
                target,
                static_gt
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
# MAIN
# =========================================================
print("=== TRAINING VISTADETR ===")
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