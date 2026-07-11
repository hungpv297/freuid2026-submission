#!/usr/bin/env python3
"""
FREUID Challenge 2026 — inference entrypoint.

Reads a FLAT directory of images from --data (default /data), runs the final
model (a 3-checkpoint ensemble of maxvit_base_tf_512, each with dense-crop
test-time augmentation), and writes a fraud score per image to
--out (default /submissions/submission.csv).

Contract (per reproducibility spec):
  * input : flat dir of images only (.jpeg/.jpg/.png/.webp/.bmp/.tif/.tiff),
            row id = filename WITHOUT extension.
  * output: CSV `id,label` — one row per image, label = float fraud score
            (higher = more confident the document is fraudulent).
  * NO network access is required or performed (pretrained=False; weights are
    loaded from local checkpoint files baked into the image).
"""
import os, sys, glob, argparse, warnings
warnings.simplefilter("ignore")
import numpy as np, cv2, torch, timm
cv2.setNumThreads(1)
from torch.utils.data import Dataset, DataLoader

MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD  = np.array((0.229, 0.224, 0.225), np.float32)
EXTS = ("*.jpeg", "*.jpg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")

# Dense-crop TTA: native + a 3x3 grid of crops taken from a 1.15x up-scaled
# image, each shown in the 4 orientations of the D2 group (identity, h-flip,
# v-flip, 180-rotate) -> 10 * 4 = 40 views, averaged in probability space.
GRID = [(gy, gx) for gy in range(3) for gx in range(3)]
FLIPS = {
    "id":   lambda x: x,
    "hf":   lambda x: torch.flip(x, [3]),
    "vf":   lambda x: torch.flip(x, [2]),
    "r180": lambda x: torch.rot90(x, 2, [2, 3]),
}


def to_tensor(rgb):
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


def make_view(im, kind, sz):
    if kind == "native":
        return cv2.resize(im, (sz, sz), interpolation=cv2.INTER_LINEAR)
    gy, gx = kind
    big = int(round(sz * 1.15)); off = big - sz
    r = cv2.resize(im, (big, big), interpolation=cv2.INTER_LINEAR)
    y = off * gy // 2; x = off * gx // 2
    return r[y:y + sz, x:x + sz]


class ImageDS(Dataset):
    """Returns one CROP variant of every image (native or a grid crop)."""
    def __init__(self, paths, kind, sz):
        self.paths, self.kind, self.sz = paths, kind, sz

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:  # unreadable -> neutral grey so we still emit a row
            im = np.full((self.sz, self.sz, 3), 128, np.uint8)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        rid = os.path.splitext(os.path.basename(p))[0]
        return to_tensor(make_view(im, self.kind, self.sz)), rid


def load_models(weights_dir, device):
    ckpts = sorted(glob.glob(os.path.join(weights_dir, "*.pt")))
    if not ckpts:
        sys.exit(f"[infer] no *.pt checkpoints found in {weights_dir}")
    models, imgsz = [], None
    for cp in ckpts:
        ck = torch.load(cp, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]; imgsz = cfg["imgsz"]
        m = timm.create_model(cfg["model"], pretrained=False, num_classes=1)
        m.load_state_dict(ck["model"])
        m = m.to(device).eval().to(memory_format=torch.channels_last)
        models.append(m)
        print(f"[infer] loaded {os.path.basename(cp)} ({cfg['model']} @ {imgsz})", flush=True)
    return models, imgsz


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/data")
    ap.add_argument("--out", default="/submissions/submission.csv")
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(__file__), "weights"))
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    paths = []
    for e in EXTS:
        paths += glob.glob(os.path.join(a.data, e))
        paths += glob.glob(os.path.join(a.data, e.upper()))
    paths = sorted(set(paths))
    if not paths:
        sys.exit(f"[infer] no images found in {a.data}")
    print(f"[infer] {len(paths)} images | device={device}", flush=True)

    models, sz = load_models(a.weights, device)

    # Accumulate summed probability over all (crop-view x flip x model) forwards.
    kinds = ["native"] + GRID
    n_views = len(kinds) * len(FLIPS) * len(models)
    acc = None; ids_ref = None
    amp = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", dtype=torch.bfloat16)
    for kind in kinds:
        dl = DataLoader(ImageDS(paths, kind, sz), batch_size=a.bs, shuffle=False,
                        num_workers=a.workers, pin_memory=(device == "cuda"))
        chunk_probs = []; chunk_ids = []
        for x, rid in dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with amp:
                s = torch.zeros(x.size(0), device=device, dtype=torch.float32)
                for fn in FLIPS.values():
                    xf = fn(x)
                    for m in models:
                        s += torch.sigmoid(m(xf).squeeze(1).float())
            chunk_probs.append(s.cpu().numpy()); chunk_ids += list(rid)
        probs = np.concatenate(chunk_probs)
        if acc is None:
            acc = probs; ids_ref = chunk_ids
        else:
            acc = acc + probs  # same order (shuffle=False) across kinds
    scores = acc / n_views

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for rid, sc in zip(ids_ref, scores):
            w.writerow([rid, f"{float(sc):.6f}"])
    print(f"[infer] wrote {len(ids_ref)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
