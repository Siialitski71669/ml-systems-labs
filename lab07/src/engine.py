import copy, os, time, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def make_loaders(cfg):
    root = cfg["data"]["root"]
    tfm = transforms.Compose([ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
    train_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=tfm)
    test = datasets.FashionMNIST(root=root, train=False, download=True, transform=tfm)

    val_n = int(cfg["data"]["validation_size"])
    g = torch.Generator().manual_seed(cfg["seed"])
    perm = torch.randperm(len(train_full), generator=g).tolist()
    train_idx, val_idx = perm[val_n:], perm[:val_n]
    max_train = cfg["training"].get("max_train_samples")
    max_test = cfg["training"].get("max_test_samples")
    if max_train:
        train_idx = train_idx[:int(max_train)]
    if max_test:
        test = Subset(test, list(range(min(int(max_test), len(test)))))
    train = Subset(train_full, train_idx)
    val = Subset(train_full, val_idx)
    kw = dict(batch_size=int(cfg["data"]["batch_size"]),
              num_workers=int(cfg["data"]["num_workers"]),
              pin_memory=torch.cuda.is_available())
    return (DataLoader(train, shuffle=True, **kw),
            DataLoader(val, shuffle=False, **kw),
            DataLoader(test, shuffle=False, **kw))

def evaluate(model, loader, device):
    model.eval()
    loss_sum = correct = total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss_sum += criterion(out, y).item() * y.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
    return loss_sum / max(total, 1), correct / max(total, 1)

def train_epochs(model, loader, val_loader, device, epochs, lr, weight_decay=0.0,
                 masks=None, teacher=None, temperature=4.0, alpha=0.5):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ce = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        model.train()
        running = n = 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            student_logits = model(x)
            if teacher is None:
                loss = ce(student_logits, y)
            else:
                with torch.no_grad():
                    teacher_logits = teacher(x)
                log_student = F.log_softmax(student_logits / temperature, dim=1)
                prob_student = log_student.exp()
                log_teacher = F.log_softmax(teacher_logits / temperature, dim=1)
                kd = (prob_student * (log_student - log_teacher)).sum(dim=1).mean() * (temperature ** 2)
                loss = alpha * ce(student_logits, y) + (1.0 - alpha) * kd
            loss.backward()
            optimizer.step()
            # Re-apply masks after every update so pruned weights stay exactly zero.
            if masks:
                with torch.no_grad():
                    for parameter, mask in masks:
                        parameter.mul_(mask)
            running += loss.item() * y.size(0)
            n += y.size(0)
        val_loss, val_acc = evaluate(model, val_loader, device)
        history.append({"epoch": epoch + 1, "train_loss": running / max(n,1),
                        "val_loss": val_loss, "val_accuracy": val_acc})
        print(f"    epoch {epoch+1}/{epochs}: val_accuracy={val_acc*100:.2f}%")
    return history

def save_checkpoint(model, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

def load_checkpoint(model, path, device):
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model

def count_nonzero_parameters(model):
    total = nonzero = 0
    for p in model.parameters():
        total += p.numel()
        nonzero += torch.count_nonzero(p).item()
    return nonzero, total

def dense_size_bytes(model, path):
    save_checkpoint(model, path)
    return Path(path).stat().st_size

def sparse_format_size_bytes(model, path):
    """Serialize each tensor as shape + nonzero flat indices + values."""
    arrays = {}
    for name, tensor in model.state_dict().items():
        a = tensor.detach().cpu().numpy().reshape(-1)
        idx = np.flatnonzero(a)
        arrays[name + "__shape"] = np.asarray(tensor.shape, dtype=np.int64)
        arrays[name + "__idx"] = idx.astype(np.int32)
        arrays[name + "__val"] = a[idx]
    np.savez_compressed(path, **arrays)
    return Path(str(path) + ".npz").stat().st_size

def measure_latency(model, loader, device, warmup_batches=10, repeats=100, threads=1):
    if device.type == "cpu":
        torch.set_num_threads(int(threads))
    model.eval()
    x, _ = next(iter(loader))
    x = x[:min(x.size(0), 128)].to(device)
    with torch.no_grad():
        for _ in range(warmup_batches):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    return (elapsed / repeats) * 1000.0 / x.size(0), x.size(0)
