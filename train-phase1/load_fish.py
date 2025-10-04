import os, shutil, random
from pathlib import Path
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# >>>>> EDIT THIS <<<<<
DATA_ROOT = "/Users/fauzan/College/Skripsi/Swin-Transformer/data/Fish_Dataset/Fish_Dataset"  # folder that directly contains the species folders
FORCE_RESPLIT = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"}

def list_images(p: Path):
    return [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in IMG_EXTS]

def has_class_subdirs(p: Path) -> bool:
    return p.exists() and any(d.is_dir() for d in p.iterdir())

def create_split(root: Path, train_dir: Path, val_dir: Path):
    print("[INFO] Creating train/val split (80/20) with tiny-class handling…")
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    classes = [d for d in root.iterdir() if d.is_dir() and d.name not in ("train", "val")]
    if not classes:
        raise SystemExit(f"[ERR] No class folders under {root}")

    for cls in sorted(classes):
        imgs = list_images(cls)
        n = len(imgs)
        if n == 0:
            print(f"[WARN] {cls.name}: 0 valid images")
            continue
        random.shuffle(imgs)
        if n == 1:
            tr, va = imgs, []
        elif n == 2:
            tr, va = imgs[:1], imgs[1:]
        else:
            k = max(1, int(0.8 * n))
            tr, va = imgs[:k], imgs[k:]

        (train_dir/cls.name).mkdir(parents=True, exist_ok=True)
        (val_dir/cls.name).mkdir(parents=True, exist_ok=True)
        for f in tr: shutil.copy2(f, train_dir/cls.name/f.name)
        for f in va: shutil.copy2(f, val_dir/cls.name/f.name)
        print(f"[OK] {cls.name}: {len(tr)} train, {len(va)} val (source {n})")

def main():
    root = Path(os.path.expanduser(DATA_ROOT)).resolve()
    train_dir = root / "train"
    val_dir   = root / "val"

    need_split = FORCE_RESPLIT or not (has_class_subdirs(train_dir) and has_class_subdirs(val_dir))
    if FORCE_RESPLIT:
        shutil.rmtree(train_dir, ignore_errors=True)
        shutil.rmtree(val_dir,   ignore_errors=True)
    if need_split:
        create_split(root, train_dir, val_dir)

    img_size = 224
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # macOS tip: start with workers=0; bump to 2 later if you want
    train_ds = datasets.ImageFolder(str(train_dir), transform=train_tfms)
    val_ds   = datasets.ImageFolder(str(val_dir),   transform=val_tfms)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0)

    print("Classes:", train_ds.classes)
    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))

    # sanity batch
    imgs, labels = next(iter(train_loader))
    print("Batch:", imgs.shape, labels.shape)

if __name__ == "__main__":
    # On macOS, guard is REQUIRED when using DataLoader with workers>0
    # You can also set the start method if needed:
    # import torch.multiprocessing as mp; mp.set_start_method("spawn", force=True)
    main()