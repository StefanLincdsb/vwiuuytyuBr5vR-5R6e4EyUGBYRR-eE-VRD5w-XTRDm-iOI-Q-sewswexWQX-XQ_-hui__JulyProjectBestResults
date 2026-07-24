from __future__ import annotations

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/pkgs/')
import time
import math
import random
import json
import copy
import pickle
import threading
from collections import deque
from datetime import datetime
import re

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split


# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
class CFG:
    OUT_DIR      = "runs_full_disjoint"
    IMG_SIZE     = 224
    VAL_FRAC     = 0.06          # not used anymore, replaced by fixed 500/class
    CALIB_FRAC   = 0.06          # not used anymore
    MAX_TRAIN_PER_CLASS = None   # not used

    MODEL        = "resnet50"
    BATCH_SIZE   = 40
    MAX_EPOCHS   = 8
    PATIENCE     = 2
    LR           = 2e-4
    HEAD_LR_MULT = 10.0
    WEIGHT_DECAY = 1e-4
    HEAD_DROPOUT = 0.2
    LABEL_SMOOTHING = 0.05
    WARMUP_FRAC  = 0.10

    SEEDS        = [0, 1, 2]
    SPLIT_SEED   = 0             # not used
    NUM_WORKERS  = 4             # not used; we use custom prefetcher
    NUM_THREADS  = os.cpu_count()  # will be set later
    LOG_EVERY    = 10            # save checkpoint and log every 10 batches

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    INITIAL_MODEL = 'resnet50-11ad3fa6.pth'

    # Paths for the dataset (relative to script location)
    TRAIN_DIR = "OCT2017/train"
    VAL_DIR   = "OCT2017/val"
    TEST_DIR  = "OCT2017/test"

mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if mps_available else 'cpu')
random.seed(CFG.SPLIT_SEED)

def log(msg: str):
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(os.path.join(CFG.OUT_DIR, "train.log"), "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


# ----------------------------------------------------------------------------
# Custom Dataset that loads images from a list of paths
# ----------------------------------------------------------------------------
class ImageDataset(Dataset):
    """Loads images from a list of relative paths (relative to root_dir)."""
    def __init__(self, root_dir, paths, labels, transform=None):
        self.root_dir = root_dir
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.paths[idx])
        image = datasets.folder.default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]


class CachedImageDataset(Dataset):
    """Preloads all images into memory as tensors."""
    def __init__(self, root_dir, paths, labels, transform):
        self.labels = labels
        self.images = []
        for p in paths:
            img_path = os.path.join(root_dir, p)
            img = datasets.folder.default_loader(img_path)
            img = transform(img)
            self.images.append(img)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


# ----------------------------------------------------------------------------
# Data loading: pickled lists or build from folders
# ----------------------------------------------------------------------------
def get_data_lists():
    """Returns (train_paths, val_paths, calib_paths, test_paths, class_names)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pickle_files = {
        'train': os.path.join(CFG.OUT_DIR, 'train.pkl'),
        'val':   os.path.join(CFG.OUT_DIR, 'val.pkl'),
        'calib': os.path.join(CFG.OUT_DIR, 'calib.pkl'),
        'test':  os.path.join(CFG.OUT_DIR, 'test.pkl'),
    }

    # Fixed class names (no longer read from disk)
    classes = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

    # If all pickle files exist, load them
    if all(os.path.exists(p) for p in pickle_files.values()):
        log("Loading pre‑saved pickle lists.")
        with open(pickle_files['train'], 'rb') as f:
            train_paths = pickle.load(f)
        with open(pickle_files['val'], 'rb') as f:
            val_paths = pickle.load(f)
        with open(pickle_files['calib'], 'rb') as f:
            calib_paths = pickle.load(f)
        with open(pickle_files['test'], 'rb') as f:
            test_paths = pickle.load(f)
        # Return the fixed class list (not the directory listing)
        return train_paths, val_paths, calib_paths, test_paths, classes

    # Otherwise, build from folders
    log("Building data lists from folder structure.")
    root = script_dir
    train_dir = os.path.join(root, CFG.TRAIN_DIR)
    val_dir   = os.path.join(root, CFG.VAL_DIR)
    test_dir  = os.path.join(root, CFG.TEST_DIR)

    class_to_idx = {c: i for i, c in enumerate(classes)}

    def get_paths_labels(dir_path):
        paths, labels = [], []
        for cls in classes:
            cls_dir = '/'.join([dir_path, cls])
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                # Skip hidden files (e.g., .DS_Store)
                if fname.startswith('.'):
                    continue
                if fname.lower().endswith(('.jpeg', '.jpg', '.png')):
                    rel_path = os.path.relpath('/'.join([cls_dir, fname]), root)
                    paths.append(rel_path)
                    labels.append(class_to_idx[cls])
        return paths, labels

    # Test list = images from test + val folders (only the four classes)
    test_paths, test_labels = get_paths_labels(test_dir)
    val_paths_tmp, val_labels_tmp = get_paths_labels(val_dir)
    test_paths.extend(val_paths_tmp)
    test_labels.extend(val_labels_tmp)
    combined = list(zip(test_paths, test_labels))
    random.shuffle(combined)
    test_paths, test_labels = zip(*combined) if combined else ([], [])
    test_paths = list(test_paths)
    test_labels = list(test_labels)

    # Get subject ids from test paths
    test_subject_ids = set()
    for p in test_paths:
        fname = os.path.basename(p)
        nums = re.findall(r'\d+', fname)
        if len(nums) >= 2:
            test_subject_ids.add(nums[-2])

    # Train list - filter out subjects that appear in test
    all_train_paths, all_train_labels = get_paths_labels(train_dir)
    filtered_train_paths = []
    filtered_train_labels = []
    for p, lab in zip(all_train_paths, all_train_labels):
        fname = os.path.basename(p)
        nums = re.findall(r'\d+', fname)
        if len(nums) >= 2 and nums[-2] in test_subject_ids:
            continue
        filtered_train_paths.append(p)
        filtered_train_labels.append(lab)

    train_paths = filtered_train_paths
    train_labels = filtered_train_labels

    # Build balanced val and calib: 500 per class each
    val_paths, val_labels = [], []
    calib_paths, calib_labels = [], []
    remaining_paths, remaining_labels = [], []

    for cls in classes:
        idx = class_to_idx[cls]
        cls_indices = [i for i, lab in enumerate(train_labels) if lab == idx]
        random.shuffle(cls_indices)
        val_indices = cls_indices[:500]
        calib_indices = cls_indices[500:1000]
        remain_indices = cls_indices[1000:]

        for i in val_indices:
            val_paths.append(train_paths[i])
            val_labels.append(train_labels[i])
        for i in calib_indices:
            calib_paths.append(train_paths[i])
            calib_labels.append(train_labels[i])
        for i in remain_indices:
            remaining_paths.append(train_paths[i])
            remaining_labels.append(train_labels[i])

    # Shuffle remaining training list
    combined_train = list(zip(remaining_paths, remaining_labels))
    for _ in range(2500):
        random.shuffle(combined_train)
    train_paths, train_labels = zip(*combined_train) if combined_train else ([], [])
    train_paths = list(train_paths)
    train_labels = list(train_labels)

    # Save as pickle files
    log("Saving pickle lists.")
    with open(pickle_files['train'], 'wb') as f:
        pickle.dump(train_paths, f)
    with open(pickle_files['val'], 'wb') as f:
        pickle.dump(val_paths, f)
    with open(pickle_files['calib'], 'wb') as f:
        pickle.dump(calib_paths, f)
    with open(pickle_files['test'], 'wb') as f:
        pickle.dump(test_paths, f)

    return train_paths, val_paths, calib_paths, test_paths, classes


# ----------------------------------------------------------------------------
# Background prefetcher for training data
# ----------------------------------------------------------------------------
class BackgroundGenerator(threading.Thread):
    """Prefetches batches in a background thread."""
    def __init__(self, generator, max_prefetch=1):
        super().__init__()
        self.queue = deque()
        self.generator = generator
        self.max_prefetch = max_prefetch
        self.daemon = True
        self.stop_event = threading.Event()
        self.start()

    def run(self):
        for item in self.generator:
            if self.stop_event.is_set():
                break
            while len(self.queue) >= self.max_prefetch:
                time.sleep(0.01)
            self.queue.append(item)
        # Signal end
        self.queue.append(None)

    def next(self):
        while len(self.queue) == 0:
            time.sleep(0.01)
        item = self.queue.popleft()
        if item is None:
            raise StopIteration
        return item

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def stop(self):
        self.stop_event.set()


def prefetched_loader(dataset, batch_size, shuffle=False, num_workers=0):
    """Returns an iterator that yields batches with background prefetching."""
    # We use a simple DataLoader with num_workers=0 to avoid multiprocessing issues
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=0, drop_last=False)
    return BackgroundGenerator(loader)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def make_head(in_features, num_classes, dropout=0.0):
    if dropout and dropout > 0:
        return nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return nn.Linear(in_features, num_classes)


def build_model(num_classes, freeze_backbone=False, head_dropout=0.0):
    m = models.resnet50(pretrained=False)
    try:
        state_dict = torch.load(CFG.INITIAL_MODEL, map_location=DEVICE)
        m.load_state_dict(state_dict)
    except:
        print(f"Warning: Could not load weights from {CFG.INITIAL_MODEL}")
    m.fc = make_head(m.fc.in_features, num_classes, head_dropout)
    m.transfer_head = m.fc
    if freeze_backbone:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.fc.parameters():
            p.requires_grad = True
    return m.to(DEVICE)


def param_groups(model, base_lr, head_lr_mult):
    head_ids = {id(p) for p in model.transfer_head.parameters()}
    backbone, head = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (head if id(p) in head_ids else backbone).append(p)
    groups = []
    if backbone: groups.append({"params": backbone, "lr": base_lr})
    if head:     groups.append({"params": head, "lr": base_lr * head_lr_mult})
    return groups


def make_optimizer_scheduler(model, steps_per_epoch):
    groups = param_groups(model, CFG.LR, CFG.HEAD_LR_MULT)
    optimizer = torch.optim.AdamW(groups, weight_decay=CFG.WEIGHT_DECAY)
    total = max(1, steps_per_epoch * CFG.MAX_EPOCHS)
    warmup = int(CFG.WARMUP_FRAC * total)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss_sum += criterion(out, y).item() * x.size(0)
        correct  += (out.argmax(1) == y).sum().item()
        n        += x.size(0)
    return loss_sum / n, correct / n


@torch.no_grad()
def predict_logits(model, loader):
    model.eval(); ys, zs = [], []
    for x, y in loader:
        zs.append(model(x.to(DEVICE)).cpu().numpy()); ys.append(y.numpy())
    return np.concatenate(ys), np.concatenate(zs)


# ----------------------------------------------------------------------------
# Checkpoint save/load
# ----------------------------------------------------------------------------
def checkpoint_path(seed, kind):
    return os.path.join(CFG.OUT_DIR, f"checkpoint_s{seed}_{kind}.pth")

def best_model_path(seed, kind):
    return os.path.join(CFG.OUT_DIR, f"best_s{seed}_{kind}.pth")

def save_checkpoint(seed, kind, model, optimizer, scheduler, epoch, batch_idx, best_val_acc, history):
    ckpt = {
        'seed': seed,
        'kind': kind,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'batch_idx': batch_idx,
        'best_val_acc': best_val_acc,
        'history': history,
    }
    torch.save(ckpt, checkpoint_path(seed, kind))

def load_checkpoint(seed, kind, model, optimizer, scheduler):
    path = checkpoint_path(seed, kind)
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    return ckpt['epoch'], ckpt['batch_idx'], ckpt['best_val_acc'], ckpt['history']


# ----------------------------------------------------------------------------
# Training one configuration (seed + kind)
# ----------------------------------------------------------------------------
def train_one(model, train_dataset, val_dataset, criterion, seed, kind, class_weights_np):
    """Train model with given seed and kind ('ft' or 'lp').
    Returns history dict and saves outputs."""
    out_npz = os.path.join(CFG.OUT_DIR, f"seed{seed}_{kind}.npz")
    if os.path.exists(out_npz):
        log(f"seed {seed} [{kind}] already done -> skipping")
        return

    # Prepare data loaders
    # Training: use prefetched loader with no shuffle (dataset already shuffled)
    # train_loader = prefetched_loader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
    # Validation: use cached dataset for speed
    val_loader = DataLoader(val_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)

    optimizer, scheduler = make_optimizer_scheduler(model, len(train_dataset) // CFG.BATCH_SIZE + 1)

    # Resume from checkpoint if exists
    start_epoch = 1
    start_batch = 0
    best_val_acc = -1.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_state = None

    ckpt_data = load_checkpoint(seed, kind, model, optimizer, scheduler)
    if ckpt_data is not None:
        start_epoch, start_batch, best_val_acc, history = ckpt_data
        log(f"Resuming seed {seed} [{kind}] from epoch {start_epoch}, batch {start_batch}")
        # If we resume in the middle of an epoch, we need to skip already processed batches
        # We'll handle this by iterating the dataset and skipping batches
    else:
        log(f"Starting fresh seed {seed} [{kind}]")

    patience = 0
    best_val_acc_sofar = best_val_acc

    for ep in range(start_epoch, CFG.MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        loss_sum, correct, n = 0.0, 0, 0
        nb = len(train_dataset) // CFG.BATCH_SIZE + (1 if len(train_dataset) % CFG.BATCH_SIZE != 0 else 0)

        train_loader = prefetched_loader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False)
        # If resuming in middle of epoch, we need to skip batches before start_batch
        batch_iter = enumerate(train_loader)
        if ep == start_epoch and start_batch > 0:
            # Skip batches until start_batch
            for bi in range(start_batch):
                try:
                    next(batch_iter)
                except StopIteration:
                    break
        else:
            start_batch = 0

        for bi, (x, y) in batch_iter:
            bi_global = start_batch + bi + 1  # 1-indexed batch number in this epoch
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)

            # Log and save checkpoint every LOG_EVERY batches
            if bi_global % CFG.LOG_EVERY == 0 or bi_global == nb:
                elapsed = time.time() - t0
                epoch_acc = correct / n if n > 0 else 0
                epoch_loss = loss_sum / n if n > 0 else 0
                log(f"    [{kind}] seed {seed} epoch {ep}/{CFG.MAX_EPOCHS} batch {bi_global}/{nb} | "
                    f"train acc {epoch_acc:.4f} loss {epoch_loss:.3f} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{elapsed:.1f}s")
                # Save checkpoint (current model state)
                save_checkpoint(seed, kind, model, optimizer, scheduler, ep, bi_global,
                                best_val_acc_sofar, history)

        # End of epoch: compute training metrics
        tr_loss = loss_sum / n if n > 0 else 0
        tr_acc = correct / n if n > 0 else 0

        # Validate (using cached validation set)
        va_loss, va_acc = evaluate(model, val_loader, criterion)
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(float(tr_loss))
        history["train_acc"].append(float(tr_acc))
        history["val_loss"].append(float(va_loss))
        history["val_acc"].append(float(va_acc))
        history["lr"].append(float(cur_lr))

        log(f"    [{kind}] seed {seed} epoch {ep}/{CFG.MAX_EPOCHS} | train {tr_acc:.4f} "
            f"(loss {tr_loss:.3f}) | val {va_acc:.4f} (loss {va_loss:.3f}) | "
            f"lr {cur_lr:.2e} | {time.time()-t0:.0f}s")

        # Check for new best validation accuracy
        if va_acc > best_val_acc_sofar + 1e-6:
            best_val_acc_sofar = va_acc
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
            # Save best model immediately
            torch.save(best_state, best_model_path(seed, kind))
            log(f"    -> new best val acc {va_acc:.4f}, model saved")
        else:
            patience += 1
            if patience >= CFG.PATIENCE:
                log(f"    [{kind}] seed {seed} early stop at epoch {ep} (best val {best_val_acc_sofar:.4f})")
                break

        # Reset start_batch for next epoch
        start_batch = 0

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        # If no improvement, just use last state
        best_state = copy.deepcopy(model.state_dict())
    try:
        os.remove(os.path.join(CFG.OUT_DIR, f'checkpoint_s{seed}_{kind}.pth'))
    except:
        pass

    # Final evaluation on test and calibration sets
    # We need test and calib datasets; they are not passed here, so we'll handle in run_config
    # Actually, we need to return the model and history, and do evaluation in run_config
    # So we'll just return the model and history, and the best state
    history["best_val_acc"] = float(best_val_acc_sofar)
    history["stopped_epoch"] = len(history["train_acc"])
    # open(os.path.join(CFG.OUT_DIR, f'ok_s{seed}_{kind}'), 'w')
    return model, history, best_state


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def run_config(seed, kind, ds, criterion, class_weights_np):
    """kind in {'ft','lp'}. Saves runs_full/seed{seed}_{kind}.npz. Resumable."""
    out_npz = os.path.join(CFG.OUT_DIR, f"seed{seed}_{kind}.npz")
    if os.path.exists(out_npz):
    # if os.path.exists(os.path.join(CFG.OUT_DIR, f'ok_s{seed}_{kind}')):
        log(f"seed {seed} [{kind}] already done -> skipping")
        return

    train_paths, val_paths, calib_paths, test_paths, classes = ds
    num_classes = len(classes)
    root_dir = os.path.dirname(os.path.abspath(__file__))

    # Build datasets
    train_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(CFG.IMAGENET_MEAN, CFG.IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(CFG.IMAGENET_MEAN, CFG.IMAGENET_STD),
    ])

    # Build class_to_idx
    class_to_idx = {c: i for i, c in enumerate(classes)}

    def get_label(path):
        # path is relative to root_dir, e.g., "OCT2017/train/CNV/xxx.jpeg"
        # Class name is the third component (0-indexed: 0=OCT2017, 1=train, 2=class)
        parts = path.split(os.sep)
        # Depending on the path structure, the class might be at index 2 if root is script_dir
        # Since paths are relative to script_dir, and we have CFG.TRAIN_DIR = "OCT2017/train"
        # So a typical path: "OCT2017/train/CNV/xxx.jpeg"
        # So class is at index 2.
        # But to be safe, we can find the class by checking which class name is in the path.
        # Simpler: the parent directory of the file is the class.
        # We can use os.path.dirname to get the directory containing the image, then basename.
        parent = os.path.basename(os.path.dirname(path))
        return class_to_idx[parent]

    # Build datasets
    train_dataset = ImageDataset(root_dir, train_paths, [get_label(p) for p in train_paths], transform=train_tf)
    # Validation set: cached for speed
    val_dataset = CachedImageDataset(root_dir, val_paths, [get_label(p) for p in val_paths], transform=eval_tf)
    calib_dataset = CachedImageDataset(root_dir, calib_paths, [get_label(p) for p in calib_paths], transform=eval_tf)
    test_dataset = CachedImageDataset(root_dir, test_paths, [get_label(p) for p in test_paths], transform=eval_tf)

    set_seed(seed)
    model = build_model(num_classes, freeze_backbone=(kind == "lp"),
                        head_dropout=CFG.HEAD_DROPOUT)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"seed {seed} [{kind}] start | {n_tr/1e6:.3f}M trainable params")

    t0 = time.time()
    model, history, best_state = train_one(model, train_dataset, val_dataset, criterion, seed, kind, class_weights_np)

    # Evaluate on test and calibration sets
    test_loader = DataLoader(test_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)
    calib_loader = DataLoader(calib_dataset, batch_size=CFG.BATCH_SIZE, shuffle=False, num_workers=0)

    test_labels, test_logits   = predict_logits(model, test_loader)
    calib_labels, calib_logits = predict_logits(model, calib_loader)
    test_pred = test_logits.argmax(1)
    acc = float((test_pred == test_labels).mean())
    log(f"seed {seed} [{kind}] DONE | test acc {acc:.4f} | "
        f"{(time.time()-t0)/60:.1f} min")

    # persist everything downstream needs
    np.savez(out_npz,
             test_labels=test_labels, test_pred=test_pred, test_logits=test_logits,
             calib_labels=calib_labels, calib_logits=calib_logits,
             class_weights=class_weights_np, classes=np.array(classes),
             history_json=np.array(json.dumps(history)))
    # also stash the best weights for the fine-tune models
    if kind == "ft":
        torch.save(best_state, os.path.join(CFG.OUT_DIR, f"seed{seed}_ft.pt"))
    else:
        torch.save(best_state, os.path.join(CFG.OUT_DIR, f"seed{seed}_lp.pt"))


def main():
    os.makedirs(CFG.OUT_DIR, exist_ok=True)
    CFG.NUM_THREADS = os.cpu_count()
    torch.set_num_threads(CFG.NUM_THREADS)
    log("=" * 70)
    log(f"train_full START | device={DEVICE} | torch={torch.__version__} | "
        f"threads={torch.get_num_threads()}")
    log(f"config: model={CFG.MODEL}, seeds={CFG.SEEDS}, max_epochs={CFG.MAX_EPOCHS}, "
        f"patience={CFG.PATIENCE}")

    # Get data lists (pickle or build)
    train_paths, val_paths, calib_paths, test_paths, classes = get_data_lists()
    num_classes = len(classes)
    log(f"classes: {classes}")
    log(f"train: {len(train_paths)} images")
    log(f"val:   {len(val_paths)} images")
    log(f"calib: {len(calib_paths)} images")
    log(f"test:  {len(test_paths)} images")

    # Class weights based on training set (inverse frequency, mean-normalized)
    root_dir = os.path.dirname(os.path.abspath(__file__))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    def get_label(path):
        parent = os.path.basename(os.path.dirname(path))
        return class_to_idx[parent]
    train_labels = [get_label(p) for p in train_paths]
    tr_counts = np.array([train_labels.count(c) for c in range(num_classes)], dtype=float)
    w = tr_counts.sum() / (num_classes * tr_counts)
    w = w / w.mean()
    class_weights = torch.tensor(w, dtype=torch.float32, device=DEVICE)
    log(f"class weights: {{{', '.join(f'{classes[c]}: {w[c]:.3f}' for c in range(num_classes))}}}")
    criterion = nn.CrossEntropyLoss(weight=class_weights,
                                    label_smoothing=CFG.LABEL_SMOOTHING)

    ds = (train_paths, val_paths, calib_paths, test_paths, classes)

    for seed in CFG.SEEDS:
        for kind in ("lp", "ft"):  # first lp (freeze backbone), then ft
            run_config(seed, kind, ds, criterion, w.astype(np.float32))

    log("ALL CONFIGS COMPLETE. Run analyze_full.py to aggregate + regenerate figures.")


if __name__ == "__main__":
    main()