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

Each image is DECODED ONCE and turned into all 40 TTA views in the dataset, so
inference scales to large private sets (no 10x re-decoding).
"""
import os, sys, glob, argparse, warnings
warnings.simplefilter("ignore")
import numpy as np, cv2, torch, timm
cv2.setNumThreads(1)
from torch.utils.data import Dataset, DataLoader

MEAN = np.array((0.485, 0.456, 0.406), np.float32)
STD  = np.array((0.229, 0.224, 0.225), np.float32)
EXTS = ("*.jpeg", "*.jpg", "*.png", "*.webp", "*.bmp", "*.tif", "*.tiff")

# Dense-crop TTA views: native 512 resize + a 3x3 grid of crops taken from a
# 1.15x up-scaled image (9 crops) -> 10 crop views. Each is later shown in the
# 4 orientations of the D2 group on-GPU -> 10 * 4 = 40 views per checkpoint.
KINDS = ["native"] + [(gy, gx) for gy in range(3) for gx in range(3)]
FLIPS = (
    lambda x: x,
    lambda x: torch.flip(x, [3]),          # h-flip
    lambda x: torch.flip(x, [2]),          # v-flip
    lambda x: torch.rot90(x, 2, [2, 3]),   # 180
)


def make_view(im_native, im_big, kind, sz):
    if kind == "native":
        return im_native
    gy, gx = kind
    off = im_big.shape[0] - sz
    y = off * gy // 2; x = off * gx // 2
    return im_big[y:y + sz, x:x + sz]


class ImageDS(Dataset):
    """Decode each image ONCE, return all KINDS crop-views stacked."""
    def __init__(self, paths, sz):
        self.paths, self.sz = paths, sz
        self.big = int(round(sz * 1.15))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        p = self.paths[i]
        im = cv2.imread(p, cv2.IMREAD_COLOR)
        if im is None:                      # unreadable -> neutral grey row
            im = np.full((self.sz, self.sz, 3), 128, np.uint8)
        else:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        native = cv2.resize(im, (self.sz, self.sz), interpolation=cv2.INTER_LINEAR)
        big = cv2.resize(im, (self.big, self.big), interpolation=cv2.INTER_LINEAR)
        # Return raw uint8 CHW views (small -> fits Docker's default 64MB /dev/shm
        # even with worker processes). Normalisation happens on-device below.
        views = [np.ascontiguousarray(make_view(native, big, k, self.sz).transpose(2, 0, 1)) for k in KINDS]
        rid = os.path.splitext(os.path.basename(p))[0]
        return torch.from_numpy(np.stack(views)), rid   # uint8 [nkind, 3, sz, sz]


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
    ap.add_argument("--weights", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights"))
    ap.add_argument("--bs", type=int, default=4, help="images per batch (each expands to len(KINDS) views)")
    ap.add_argument("--workers", type=int, default=2, help="dataloader workers (default 2 fits Docker's 64MB /dev/shm; raise with --shm-size)")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    mean_t = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std_t = torch.tensor(STD, device=device).view(1, 3, 1, 1)

    paths = []
    for e in EXTS:
        paths += glob.glob(os.path.join(a.data, e))
        paths += glob.glob(os.path.join(a.data, e.upper()))
    paths = sorted(set(paths))
    if not paths:
        sys.exit(f"[infer] no images found in {a.data}")
    print(f"[infer] {len(paths)} images | device={device}", flush=True)

    models, sz = load_models(a.weights, device)
    nk = len(KINDS); denom = float(nk * len(FLIPS) * len(models))
    dl = DataLoader(ImageDS(paths, sz), batch_size=a.bs, shuffle=False,
                    num_workers=a.workers, pin_memory=(device == "cuda"))
    amp = (torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda"
           else torch.autocast("cpu", dtype=torch.bfloat16))

    ids_out, scores_out = [], []
    for views, rid in dl:                    # uint8 views [B, nk, 3, sz, sz]
        B = views.size(0)
        x = views.view(B * nk, 3, sz, sz).to(device, non_blocking=True).float().div_(255.0)
        x = ((x - mean_t) / std_t).to(memory_format=torch.channels_last)
        with amp:
            s = torch.zeros(B * nk, device=device, dtype=torch.float32)
            for fn in FLIPS:
                xf = fn(x)
                for m in models:
                    s += torch.sigmoid(m(xf).squeeze(1).float())
        s = s.view(B, nk).sum(1) / denom     # mean over views * flips * models
        scores_out.append(s.cpu().numpy()); ids_out += list(rid)
    scores = np.concatenate(scores_out)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    import csv
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "label"])
        for rid, sc in zip(ids_out, scores):
            w.writerow([rid, f"{float(sc):.6f}"])
    print(f"[infer] wrote {len(ids_out)} rows -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
