"""
mnms2_densenet161.py

Extension of NeSyFOLD to:
  - DenseNet-161 backbone instead of VGG16
  - M&Ms-2 cardiomyopathy classification dataset

Assumptions:
  - You have two CSVs: train and test, both with columns: SUBJECT_CODE,DISEASE
  - A validation split is created internally from the train set.

Example usage:

  python mnms2_densenet161.py \
      --train_csv /path/to/mnms2_train.csv \
      --test_csv  /path/to/mnms2_test.csv \
      --img_root  /path/to/mnms2_root \
      --model_check_dir /path/to/checkpoints \
      --foldsem_user you@example.com \
      --foldsem_password YOUR_PASSWORD
"""

import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F  # kept for possible extensions
import torchvision.transforms as T
from torchvision.models import densenet161
from torchvision.models._utils import IntermediateLayerGetter
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision.models as tv_models

import nibabel as nib
import random
import requests
import config  # for foldsem_user and foldsem_pass  

from simplify_rules import simplify_rule  # from nesyfold_codebase
import algo  # for NeSyFOLD filter table creation (adapted for DenseNet-161)
from foldsem_api import foldsem_api  # for calling Fold-SE-M multicategory API (adapted for DenseNet-161)

def get_fidelity(data_loader, model, y_f,device):
    model.eval()
    f = 0
    # calculate the total test set accuracy
    with torch.no_grad():
        y_m = []
        for batch_idx, (inputs, targets) in enumerate(tqdm(data_loader)):
            inputs, targets = inputs.float().to(device), targets.to(device)
            ypred = model(inputs)
            values, indices = torch.max(ypred, 1)
            ypred_list = indices.tolist()
            y_m.extend(ypred_list)
    #calculate the accuracy between y_train_m and y_train_f
    y_m = [str(i) for i in y_m]
    for i in range(len(y_m)):
        if y_m[i] == y_f[i]:
            f += 1
    f = f/len(y_m)
    return f

# -----------------------
# M&Ms-2 Dataset wrapper
# -----------------------

CARDIOMYOPATHY_LABELS = ['NOR', 'LV', 'HCM', 'ARR', 'FALL', 'CIA', 'RV', 'TRI']
LABEL_TO_IDX = {lab: i for i, lab in enumerate(CARDIOMYOPATHY_LABELS)}

def pad_subject_code(code) -> str:
    # Ensure 3-digit SUBJECT_CODE, e.g. 1 -> "001"
    return str(code).zfill(3)   # 1 -> "001", 12 -> "012", 123 -> "123"[web:75]


class MnMs2MultiViewNiftiDataset(Dataset):
    """
    Reads M&Ms-2 per-ID folders with the 10 files:

      ID_LA_CINE.nii.gz
      ID_LA_ED_gt_.nii.gz
      ID_LA_ED.nii.gz
      ID_LA_ES_gt_.nii.gz
      ID_LA_ES.nii.gz
      ID_SA_CINE.nii.gz
      ID_SA_ED_gt_.nii.gz
      ID_SA_ED.nii.gz
      ID_SA_ES_gt_.nii.gz
      ID_SA_ES.nii.gz

    CSV is expected to have columns: SUBJECT_CODE, DISEASE

    view_phase_list: list of strings like ["LA_ED", "SA_ED", "LA_ES", "SA_ES"].
    For each sample we pick one (view, phase) from this list.

    class_list: optional list of disease codes, e.g. ["NOR", "HCM"].
      If provided, the dataset is filtered to those diseases and labels
      are remapped to 0..len(class_list)-1.
      If None, all CARDIOMYOPATHY_LABELS present in the CSV are used.
    """

    def __init__(
        self,
        csv_path: str,
        img_root: str,
        view_phase_list= None,
        transform=None,
        class_list = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.transform = transform

        if not {"SUBJECT_CODE", "DISEASE"}.issubset(self.df.columns):
            raise ValueError("CSV must contain SUBJECT_CODE and DISEASE columns.")

        # Validate labels against global superset
        all_labels = set(self.df["DISEASE"])
        unknown = all_labels - set(CARDIOMYOPATHY_LABELS)
        if unknown:
            raise ValueError(f"Unexpected labels found in DISEASE column: {unknown}")

        # Handle class subset selection
        if class_list is None or len(class_list) == 0:
            # Use all classes present in the CSV (intersection with global list, to preserve order)
            active_classes = [c for c in CARDIOMYOPATHY_LABELS if c in all_labels]
        else:
            # Normalize user-specified classes and validate
            class_list_norm = [c.strip().upper() for c in class_list]
            bad = set(class_list_norm) - set(CARDIOMYOPATHY_LABELS)
            if bad:
                raise ValueError(f"class_list contains unknown classes: {bad}")
            active_classes = class_list_norm

        # Filter dataframe to the selected classes only
        self.df = self.df[self.df["DISEASE"].isin(active_classes)].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError("After filtering by class_list, no samples remain.")

        # Store class information and local mapping
        self.classes = active_classes
        self.label_to_idx = {lab: i for i, lab in enumerate(self.classes)}

        # Views / phases
        if view_phase_list is None or len(view_phase_list) == 0:
            view_phase_list = ["LA_ED"]

        self.configs = []
        for vp in view_phase_list:
            vp_norm = vp.strip().upper()
            if "_" not in vp_norm:
                raise ValueError(f"view_phase '{vp}' must be of form 'LA_ED' or 'SA_ES'")
            view, phase = vp_norm.split("_", 1)
            if view not in {"LA", "SA"}:
                raise ValueError(f"View must be 'LA' or 'SA', got '{view}'")
            if phase not in {"ED", "ES"}:
                raise ValueError(f"Phase must be 'ED' or 'ES', got '{phase}'")
            self.configs.append((view, phase))

    def __len__(self):
        return len(self.df)

    def _volume_path(self, case_id: str, view: str, phase: str) -> str:
        filename = f"{case_id}_{view}_{phase}.nii.gz"
        return os.path.join(self.img_root, case_id, filename)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case_id = pad_subject_code(row["SUBJECT_CODE"])
        label_str = row["DISEASE"]
        label = self.label_to_idx[label_str]

        view, phase = random.choice(self.configs)

        vol_path = self._volume_path(case_id, view, phase)
        img_nii = nib.load(vol_path)
        vol = img_nii.get_fdata()  # typically (H, W, S)

        mid_slice_idx = vol.shape[-1] // 2
        slice2d = vol[..., mid_slice_idx]

        slice2d = slice2d - slice2d.min()
        if slice2d.max() > 0:
            slice2d = slice2d / slice2d.max()
        slice2d = (slice2d * 255).astype(np.uint8)

        im = Image.fromarray(slice2d).convert("RGB")

        if self.transform is not None:
            im = self.transform(im)

        return im, label


def get_mnms2_dataloaders(
    train_csv: str,
    test_csv: str,
    img_root: str,
    batch_size: int = 8,
    num_workers: int = 4,
    view_phase_list=None,   # e.g. ["LA_ED"] or ["LA_ED","SA_ED","LA_ES","SA_ES"]
    val_frac: float = 0.1,
    seed: int = 42,
    class_list = None  # NEW: select subset of classes
):
    """
    Build train/val/test loaders from train+test CSVs.
    A val subset is carved out from the train set with random_split.

    class_list: if not None, use only these DISEASE labels (e.g. ["NOR","HCM"]).
                If None, use all classes present in CSV.
    """
    transform = T.Compose(
        [
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Full train dataset (before val split)
    train_full_ds = MnMs2MultiViewNiftiDataset(
        train_csv,
        img_root,
        view_phase_list=view_phase_list,
        transform=transform,
        class_list=class_list,
    )

    n_full = len(train_full_ds)
    val_size = max(1, int(val_frac * n_full))
    train_size = n_full - val_size

    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(train_full_ds, [train_size, val_size], generator=gen)

    # Test dataset (same subset of classes)
    test_ds = MnMs2MultiViewNiftiDataset(
        test_csv,
        img_root,
        view_phase_list=view_phase_list,
        transform=transform,
        class_list=train_full_ds.classes,  # ensure same class subset/order
    )

    # Class weights from the train subset
    full_df = train_full_ds.df
    train_indices = train_ds.indices  # indices into full_df
    labels = [train_full_ds.label_to_idx[full_df["DISEASE"].iloc[i]] for i in train_indices]
    counts = np.bincount(labels, minlength=len(train_full_ds.classes))
    class_weights = 1.0 / np.maximum(counts, 1)
    class_weights = class_weights / class_weights.sum()
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    num_classes = len(train_full_ds.classes)
    return train_loader, val_loader, test_loader, num_classes, class_weights


# -----------------------
# DenseNet-161 backbone
# -----------------------
'''
def build_densenet161(num_classes: int) -> nn.Module:
    """
    Load DenseNet-161 pretrained on ImageNet and replace classifier for cardiomyopathy classes.
    """
    #model = densenet161(pretrained=True)
    model = densenet161(weights="IMAGENET1K_V1")  # torchvision >= 0.13
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
'''

# -----------------------
# Generic backbone builder
# -----------------------

def build_backbone(model_name: str, num_classes: int):
    """
    Build a torchvision model by name, replace its classifier with a num_classes
    output layer, and also build a feature extractor that returns the activations
    right BEFORE the final linear classifier layer.

    Returns:
        model: full classifier model
        feature_model: wrapper whose forward(x) -> features [B, F]
    """
    if not hasattr(tv_models, model_name):
        raise ValueError(f"torchvision.models has no model named '{model_name}'")

    # Instantiate pretrained backbone (fallback if signature has no 'pretrained')
    try:
        model = getattr(tv_models, model_name)(pretrained=True)
    except TypeError:
        model = getattr(tv_models, model_name)()

    # ----- replace classifier for num_classes -----
    # Case A: classifier is a single Linear
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

    # Case B: classifier is a Sequential (e.g. VGG, DenseNet, EfficientNet)
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        layers = list(model.classifier)
        for i in reversed(range(len(layers))):
            if isinstance(layers[i], nn.Linear):
                in_features = layers[i].in_features
                layers[i] = nn.Linear(in_features, num_classes)
                break
        model.classifier = nn.Sequential(*layers)

    # Case C: ResNet-style .fc
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

    else:
        # Fallback: replace last Linear anywhere in the model
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError(f"Could not find a Linear classifier head in model '{model_name}'")
        in_features = last_linear.in_features
        for name, module in model.named_modules():
            if module is last_linear:
                parent_name = ".".join(name.split(".")[:-1])
                attr_name = name.split(".")[-1]
                parent = model if parent_name == "" else dict(model.named_modules())[parent_name]
                setattr(parent, attr_name, nn.Linear(in_features, num_classes))
                break

    # ----- build feature extractor: last layer BEFORE final linear -----
    # Decide which module we want to hook
    target_module = None

    # ResNet-style: hook fc (we want its input)
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        target_module = model.fc

    # VGG/DenseNet/EfficientNet-style: classifier is Sequential
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
        last_linear = None
        for m in model.classifier:
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError("classifier is Sequential but contains no Linear layers.")
        target_module = last_linear

    # classifier is a single Linear
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        target_module = model.classifier

    # Fallback: last Linear anywhere in the model
    else:
        last_linear = None
        for m in model.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is None:
            raise ValueError("Could not find any Linear classifier module in the model.")
        target_module = last_linear

    class LastFeatureWrapper(nn.Module):
        """
        Wraps the backbone; forward(x) returns the input to target_module,
        i.e. the last representation before the final linear classifier.
        """

        def __init__(self, backbone: nn.Module, hook_module: nn.Module):
            super().__init__()
            self.backbone = backbone
            self.hook_module = hook_module
            self._feat = None

            def pre_hook(module, inputs):
                # inputs[0] is what goes into the final Linear
                self._feat = inputs[0].detach()

            self._hook = self.hook_module.register_forward_pre_hook(pre_hook)

        def forward(self, x):
            _ = self.backbone(x)
            return self._feat

    feature_model = LastFeatureWrapper(model, target_module)

    return model, feature_model

# -----------------------
# Training / evaluation
# -----------------------

def train_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(loader, desc="Train", leave=False):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    """
    Returns: (avg_loss, accuracy)
    """
    model.eval()
    running_loss = 0.0
    total_correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Eval", leave=False):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = running_loss / max(len(loader), 1)
    acc = total_correct / max(total, 1)
    return avg_loss, acc

'''
def train_densenet_for_mnms2(
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
    class_weights: torch.Tensor,
    device: str,
    checkpoints_dir: str,
    run_id: int,
    max_epochs: int = 100,
) -> nn.Module:
    """
    Train DenseNet-161 on M&Ms-2, roughly mirroring NeSyFOLD's VGG16 training setup.
    Saves checkpoints and returns the trained model.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)

    model = build_densenet161(num_classes)
    model = model.to(device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(
        model.parameters(),
        lr=5e-7,      # as in NeSyFOLD paper
        weight_decay=5e-3,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
    )

    writer = SummaryWriter(log_dir=os.path.join("train_logs_mnms2", f"run_{run_id}"))
    best_val_acc = 0.0
    best_path = os.path.join(checkpoints_dir, f"best_run_{run_id}.pt")

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoints_dir, f"chkpt_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            ckpt_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  New best val_acc={best_val_acc:.4f}  ->  saved {best_path}")

    writer.close()

    # Load best weights before returning
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    return model
'''

def train_backbone_for_mnms2(
    model_name: str,
    train_loader,
    val_loader,
    test_loader,
    num_classes: int,
    class_weights: torch.Tensor,
    device: str,
    checkpoints_dir: str,
    run_id: int,
    max_epochs: int = 100,
) -> nn.Module:
    """
    Train a generic torchvision backbone on M&Ms-2.
    Saves checkpoints into checkpoints_dir/model_name.
    """
    os.makedirs(checkpoints_dir, exist_ok=True)

    model, feature_model = build_backbone(model_name, num_classes)
    model = model.to(device)
    feature_model = feature_model.to(device)

    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = Adam(
        model.parameters(),
        lr=5e-7,
        weight_decay=5e-3,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        factor=0.5,
        patience=10,
    )

    log_dir = os.path.join("train_logs_mnms2", model_name, f"run_{run_id}")
    writer = SummaryWriter(log_dir=log_dir)
    best_val_acc = 0.0
    best_path = os.path.join(checkpoints_dir, f"best_run_{run_id}.pt")

    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch + 1}/{max_epochs}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        writer.add_scalar("acc/val", val_acc, epoch)

        print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(checkpoints_dir, f"chkpt_epoch_{epoch + 1}.pt")
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
            },
            ckpt_path,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"  New best val_acc={best_val_acc:.4f}  ->  saved {best_path}")

    writer.close()

    # Load best weights before returning
    model.load_state_dict(torch.load(best_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nFinal test_loss={test_loss:.4f}  test_acc={test_acc:.4f}")

    return model, feature_model

# -----------------------
# NeSyFOLD filter tables with DenseNet-161
# -----------------------

def create_filter_tables_densenet161(
    train_loader,
    val_loader,
    test_loader,
    model: nn.Module,
    device: str,
    out_dir: str,
    alpha: float = 0.6,
    gamma: float = 0.7,
):
    """
    DenseNet-161 version of algo.create_filter_data:

      1. collect norms of last conv features (features.norm5)
      2. compute per-kernel thresholds: theta_k = alpha * mean + gamma * std
      3. binarize (>= theta_k) to build Q tensors for train/val/test
      4. append label column (last) and save as CSV

    This generalises the original VGG16-specific code (512 features) to
    arbitrary num_kernels.
    """
    model.eval()

    # Number of kernels is the input dimension of classifier
    num_kernels = model.classifier.in_features

    # We tap the last conv feature map via IntermediateLayerGetter on model.features
    return_layers = {"norm5": "feat"}  # name 'norm5' is from torchvision DenseNet
    feat_getter = IntermediateLayerGetter(model.features, return_layers=return_layers)

    os.makedirs(out_dir, exist_ok=True)
    train_filter_path = os.path.join(out_dir, "train_filters.csv")
    val_filter_path = os.path.join(out_dir, "val_filters.csv")
    test_filter_path = os.path.join(out_dir, "test_filters.csv")

    def compute_norm_tensor(loader, split_name: str) -> torch.Tensor:
        dataset = loader.dataset
        norm_tensor = torch.empty(len(dataset), num_kernels, dtype=torch.float32)

        idx_offset = 0
        for inputs, targets in tqdm(loader, desc=f"Norms-{split_name}"):
            inputs = inputs.to(device, non_blocking=True)
            with torch.no_grad():
                feats = feat_getter(inputs)["feat"]  # [B, C, H, W]
                # L2 norm over spatial dims per kernel
                batch_norms = torch.linalg.norm(feats, ord=2, dim=(2, 3))  # [B, C]
            bsz = batch_norms.size(0)
            norm_tensor[idx_offset: idx_offset + bsz] = batch_norms.cpu()
            idx_offset += bsz

        return norm_tensor

    # 1. Train norms and thresholds
    train_norms = compute_norm_tensor(train_loader, "train")

    theta = alpha * train_norms.mean(dim=0) + gamma * train_norms.std(dim=0)

    # Binarize
    train_Q = (train_norms >= theta).to(torch.int32)

    # Append targets (train subset)
    train_targets = []
    for _, target in tqdm(train_loader.dataset, desc="Targets-train", leave=False):
        train_targets.append(target)
    train_targets = torch.tensor(train_targets, dtype=torch.int32).unsqueeze(1)

    train_Q_full = torch.cat([train_Q, train_targets], dim=1).numpy()
    train_df = pd.DataFrame(train_Q_full)
    train_df.to_csv(train_filter_path, index=False)
    print(f"Saved train filter table to {train_filter_path}")

    # 2. Val norms
    val_norms = compute_norm_tensor(val_loader, "val")
    val_Q = (val_norms >= theta).to(torch.int32)
    val_targets = []
    for _, target in tqdm(val_loader.dataset, desc="Targets-val", leave=False):
        val_targets.append(target)
    val_targets = torch.tensor(val_targets, dtype=torch.int32).unsqueeze(1)

    val_Q_full = torch.cat([val_Q, val_targets], dim=1).numpy()
    val_df = pd.DataFrame(val_Q_full)
    val_df.to_csv(val_filter_path, index=False)
    print(f"Saved val filter table to {val_filter_path}")

    # 3. Test norms
    test_norms = compute_norm_tensor(test_loader, "test")
    test_Q = (test_norms >= theta).to(torch.int32)
    test_targets = []
    for _, target in tqdm(test_loader.dataset, desc="Targets-test", leave=False):
        test_targets.append(target)
    test_targets = torch.tensor(test_targets, dtype=torch.int32).unsqueeze(1)

    test_Q_full = torch.cat([test_Q, test_targets], dim=1).numpy()
    test_df = pd.DataFrame(test_Q_full)
    test_df.to_csv(test_filter_path, index=False)
    print(f"Saved test filter table to {test_filter_path}")

    return train_filter_path, val_filter_path, test_filter_path, num_kernels


# -----------------------
# Fold-SE-M connection (multi-category, sampleCode-v1.2 style)
# -----------------------

FOLDSEM_URL_MULTICLASS = "http://ec2-52-0-60-249.compute-1.amazonaws.com/auth/foldmodel_multicategory/"


def run_foldsem_multiclass(
    train_csv_path: str,
    rule_file_path: str,
    username: str,
    password: str,
    target_col_idx: int,
    test_csv_path: str = None,
    hyp1: str = "",       # "" => only rules, no internal split
    hyp2: str = "0.5",    # exception ratio
    hyp3: str = "0.005",  # tail ratio
    save_model: str = "local",
    numattrs: str = "",   # numeric features (comma-separated) – not used here
):
    """
    Call Fold-SE-M multicategory API following the official sample code style.

    - train_csv_path: CSV with feature columns + target (last column index = target_col_idx)
    - rule_file_path: where to save simplified rules (.pl or .txt)
    - username/password: Fold-SE-M credentials
    - target_col_idx: index of the label column in the CSV (0-based)
    - test_csv_path: optional CSV with same columns for evaluation
    - hyp1/hyp2/hyp3: Fold-SE-M hyperparameters as strings
    - save_model: "local" to receive model_json, anything else to skip
    - numattrs: comma-separated numeric feature names (unused for binary filter bits)
    """

    # 1) Load train (and optional test) data
    df = pd.read_csv(train_csv_path)
    df.columns = [f"str_{i}" for i in range(len(df.columns))]

    if target_col_idx < 0 or target_col_idx >= len(df.columns):
        raise ValueError(f"target_col_idx {target_col_idx} out of range for {len(df.columns)} columns")

    target_column = df.columns[target_col_idx]

    # all other columns treated as categorical (string) attributes
    feature_cols = [c for c in df.columns if c != target_column]
    strattrs = ",".join(feature_cols)

    data_frame_json = df.to_json(orient="split")

    if test_csv_path is not None:
        test_df = pd.read_csv(test_csv_path)
        test_df.columns = df.columns
        test_data_frame_json = test_df.to_json(orient="split")
    else:
        test_data_frame_json = ""

    # 2) Build payload
    payload = {
        "username": username,
        "password": password,
        "data_frame_json": data_frame_json,
        "numattrs": numattrs,
        "strattrs": strattrs,
        "hyp1": hyp1,
        "hyp2": hyp2,
        "hyp3": hyp3,
        "positive_value": "",  # not used for multi-class
        "test_data_frame_json": test_data_frame_json,
        "label_value": target_column,
        "save_model": save_model,
    }

    # 3) POST to Fold-SE-M multicategory endpoint
    response = requests.post(FOLDSEM_URL_MULTICLASS, json=payload)
    print(f"Fold-SE-M response status: {response.status_code}")
    print(f"Fold-SE-M response text: {response.text[:200]}...")  # print first 200 chars
    # 4) Parse response
    try:
        response_obj = response.json()
    except Exception:
        print("Non-JSON response from Fold-SE-M:")
        print(response)
        return None, None, None

    try:
        if response_obj.get("error") is None:
            rules_str = response_obj.get("rules", "")
            n_rules   = response_obj.get("n_rules")
            n_preds   = response_obj.get("n_preds")
            size      = response_obj.get("size")

            # Save simplified rules to file, one per line
            with open(rule_file_path, "w") as f:
                for rule in rules_str.split("\n"):
                    rule = rule.strip()
                    if not rule:
                        continue
                    rule = rule.replace("str_", "")
                    simplified = simplify_rule(rule)
                    f.write(simplified + "\n")

            print(f"Saved rules to {rule_file_path}")
            print(f"Fold-SE-M: n_rules={n_rules}, n_preds={n_preds}, size={size}")
            return n_rules, n_preds, size
        else:
            print("Fold-SE-M error:", response_obj["error"])
    except Exception as e:
        print("Error processing Fold-SE-M response:", e)

    return None, None, None


# -----------------------
# Main
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="NeSyFOLD with DenseNet-161 on M&Ms-2 cardiomyopathy")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to M&Ms-2 train CSV (ID,Class)")
    parser.add_argument("--test_csv", type=str, required=True, help="Path to M&Ms-2 test CSV (ID,Class)")
    parser.add_argument("--img_root", type=str, required=True, help="Root directory for per-ID folders with NIfTI files")
    parser.add_argument("--model_check_dir", type=str, required=True, help="Directory to save DenseNet checkpoints")
    parser.add_argument("--filters_out_dir", type=str, default="mnms2_filters", help="Directory to save filter tables")
    #parser.add_argument("--foldsem_user", type=str, required=True, help="FOLD-SE-M username (email)")
    #parser.add_argument("--foldsem_password", type=str, required=True, help="FOLD-SE-M password")
    parser.add_argument("--ratio", type=float, default=0.5, help="FOLD-SE-M exception ratio (hyp2)")
    parser.add_argument("--tail", type=float, default=0.005, help="FOLD-SE-M tail ratio (hyp3)")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--val_frac", type=float, default=0.1, help="Fraction of train used for validation")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for train/val split")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    foldsem_user = config.foldsem_user
    foldsem_password = config.foldesem_pass

    # 1) Data
    model_name = "resnet152"  # or any other torchvision model name vgg16
    view_phase_list = ["LA_ED","SA_ED","LA_ES","SA_ES"]  # you can extend to ["LA_ED","SA_ED","LA_ES","SA_ES"]
    #LA_ED = test_acc = 0.7167 / LA_ES = test_acc = 0.7389 / LA_ED+LA_ES = test_acc = 0.7222
    #SA_ED = test_acc = 0.7444 / SA_ES = test_acc = 0.6778  / SA_ED+SA_ES = test_acc = 0.6944
    #LA_ED+SA_ED = test_acc = 0.6500 / LA_ES+SA_ES = test_acc = 0.6778
    #LA_ED+SA_ED+LA_ES+SA_ES = test_acc = 0.5722
    classes = None
    #classes=["NOR","HCM","ARR"]  # optional: select subset of classes for training
    #SA_ED + NOR+LV+RV = test_acc = 0.8657
    train_loader, val_loader, test_loader, num_classes, class_weights = get_mnms2_dataloaders(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        img_root=args.img_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        view_phase_list=view_phase_list,
        val_frac=args.val_frac,
        seed=args.seed,
        class_list=classes
    )

    # 2) Train DenseNet-161
    '''
    model = train_densenet_for_mnms2(
        train_loader,
        val_loader,
        test_loader,
        num_classes,
        class_weights,
        device,
        checkpoints_dir=args.model_check_dir,
        run_id=1,
        max_epochs=args.epochs,
    )
    '''
    
    # 2) Train backbone
    model, feature_model = train_backbone_for_mnms2(
        model_name=model_name,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_classes=num_classes,
        class_weights=class_weights,
        device=device,
        checkpoints_dir=args.model_check_dir,
        run_id=1,
        max_epochs=args.epochs,
    )

    #model = build_densenet161(num_classes)
    #model.load_state_dict(torch.load(os.path.join(args.model_check_dir, "best_run_1.pt"), map_location=device))
    #model = model.to(device)
    # 3) Create filter tables (NeSyFOLD step)
    '''
    train_filter_path, val_filter_path, test_filter_path, num_kernels = create_filter_tables_densenet161(
        train_loader,
        val_loader,
        test_loader,
        model,
        device=device,
        out_dir=args.filters_out_dir,
        alpha=0.6,
        gamma=0.7,
    )

    print(f"num_kernels (DenseNet-161 last conv channels): {num_kernels}")
    '''

    algo.create_norm_tensor_generic(train_loader, feature_model, args.model_check_dir+'/norms', device)

    params = {"batch_size": 32, "epochs": 100, "lr": 5e-7, "l2": 5e-3, "decay_factor": 0.5, "patience": 10}
    alpha = 0.6
    gamma = 0.7

    algo.create_filter_data_generic(train_loader,
    val_loader,
    test_loader,
    feature_model,
    args.model_check_dir+'/norms',
    os.path.join(args.filters_out_dir, "train_filters.csv"),
    os.path.join(args.filters_out_dir, "val_filters.csv"),
    os.path.join(args.filters_out_dir, "test_filters.csv"),
    device,
    params,
    alpha,
    gamma)


    # 4) Call Fold-SE-M to get neurosymbolic rule-set
    rule_file_path = os.path.join(args.filters_out_dir, f"rules_mnms2_{model_name}_{str(view_phase_list)}_{str(classes)}.pl")

    # target_col_idx = num_kernels (last column in filter tables is label)
    '''
    n_rules, n_preds, size = run_foldsem_multiclass(
        train_csv_path=train_filter_path,
        rule_file_path=rule_file_path,
        username=foldsem_user,
        password=foldsem_password,
        target_col_idx=num_kernels,
        test_csv_path=test_filter_path,          # optional external test for Fold-SE-M
        hyp1="",                                 # "" -> no internal train/test split, only rules
        hyp2=str(args.ratio),
        hyp3=str(args.tail),
        save_model="local",
        numattrs="",                             # all filter bits treated as categorical
    )

    print(f"Fold-SE-M returned: n_rules={n_rules}, n_preds={n_preds}, size={size}")
    print(f"Rules written to: {rule_file_path}")
    '''
    ratio = 0.8
    tail = 5e-3
    acc_train, acc_val, acc_test, y_train_f, y_val_f, y_test_f, n_rules, n_preds, size = foldsem_api(
        os.path.join(args.filters_out_dir, "train_filters.csv"), 
        os.path.join(args.filters_out_dir, "val_filters.csv"),
        os.path.join(args.filters_out_dir, "test_filters.csv"), 
        rule_file_path, 
        ratio, 
        tail, 
        config.foldsem_user, 
        config.foldesem_pass)

    # calculate fidelity
    f_train = get_fidelity(train_loader, model, y_train_f,device)
    f_val = get_fidelity(val_loader, model, y_val_f,device)
    f_test = get_fidelity(test_loader, model, y_test_f,device)

    print(f"Training Accuracy: {acc_train}")
    print(f"Validation Accuracy: {acc_val}")
    print(f"Test Accuracy: {acc_test}")
    print(f"Training Fidelity: {f_train}")
    print(f"Validation Fidelity: {f_val}")
    print(f"Test Fidelity: {f_test}")
    print(f"Fold-SE-M returned: n_rules={n_rules}, n_preds={n_preds}, size={size}")

if __name__ == "__main__":
    main()