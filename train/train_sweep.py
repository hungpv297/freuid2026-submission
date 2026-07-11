import os, sys, json, time, math, argparse, warnings, glob
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, cv2
cv2.setNumThreads(0)
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import timm
from sklearn.metrics import roc_auc_score
MEAN=(0.485,0.456,0.406); STD=(0.229,0.224,0.225)

def build_aug(name,imgsz,train):
    N=A.Normalize(MEAN,STD); T=ToTensorV2(); R=A.Resize(imgsz,imgsz)
    if not train: return A.Compose([R,N,T])                       # clean val/test pipeline always
    flips=[A.HorizontalFlip(p=0.5),A.VerticalFlip(p=0.5)]
    if name in ("v1","basic"):
        return A.Compose(flips+[A.Rotate(limit=20,border_mode=cv2.BORDER_REFLECT_101,p=0.5),R,N,T])
    if name=="v2":   # print-and-capture / recompression realism
        return A.Compose(flips+[A.Rotate(limit=15,border_mode=cv2.BORDER_REFLECT_101,p=0.5),
            A.ImageCompression(quality_range=(40,90),p=0.7),
            A.OneOf([A.GaussianBlur(blur_limit=(3,7)),A.MotionBlur(blur_limit=(3,7))],p=0.3),
            A.RandomBrightnessContrast(0.2,0.2,p=0.5),A.GaussNoise(p=0.3),
            A.Perspective(scale=(0.02,0.06),p=0.3),R,N,T])
    if name=="v3":   # photometric + color + light occlusion
        return A.Compose(flips+[A.Rotate(limit=15,border_mode=cv2.BORDER_REFLECT_101,p=0.5),
            A.RandomBrightnessContrast(0.3,0.3,p=0.6),A.HueSaturationValue(20,30,20,p=0.4),
            A.CLAHE(clip_limit=2.0,p=0.3),
            A.CoarseDropout(num_holes_range=(1,6),hole_height_range=(0.04,0.12),
                            hole_width_range=(0.04,0.12),p=0.3),R,N,T])
    if name=="v4":   # geometric heavy
        return A.Compose([A.RandomResizedCrop(size=(imgsz,imgsz),scale=(0.7,1.0),ratio=(0.75,1.33),p=1.0)]+
            flips+[A.Affine(translate_percent=(0,0.0625),scale=(0.9,1.1),rotate=(-20,20),p=0.7),
            A.Perspective(scale=(0.02,0.08),p=0.3),N,T])
    if name=="v5":   # combined moderate
        return A.Compose(flips+[A.Rotate(limit=15,border_mode=cv2.BORDER_REFLECT_101,p=0.5),
            A.ImageCompression(quality_range=(50,90),p=0.5),A.RandomBrightnessContrast(0.2,0.2,p=0.5),
            A.GaussNoise(p=0.2),
            A.CoarseDropout(num_holes_range=(1,6),hole_height_range=(0.04,0.10),
                            hole_width_range=(0.04,0.10),p=0.3),R,N,T])
    raise ValueError(name)

class DS(Dataset):
    def __init__(self,paths,labels,tf): self.p=paths; self.y=labels; self.tf=tf
    def __len__(self): return len(self.p)
    def __getitem__(self,i):
        im=cv2.cvtColor(cv2.imread(self.p[i]),cv2.COLOR_BGR2RGB)
        return self.tf(image=im)["image"], (self.y[i] if self.y is not None else os.path.splitext(os.path.basename(self.p[i]))[0])

def apcer_at_bpcer(y,s,b=0.01):
    thr=np.quantile(s[y==0],1-b); return float((s[y==1]<thr).mean())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--fold",type=int,default=0); ap.add_argument("--imgsz",type=int,default=448)
    ap.add_argument("--bs",type=int,default=24); ap.add_argument("--lr",type=float,default=2e-4)
    ap.add_argument("--epochs",type=int,default=10); ap.add_argument("--workers",type=int,default=12)
    ap.add_argument("--aug",default="v1"); ap.add_argument("--pseudo",default="")
    a=ap.parse_args()
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
    dev="cuda"; outdir=f"/mnt/ecai/exp/{a.exp}"; os.makedirs(outdir,exist_ok=True)
    def log(d): print(json.dumps(d),flush=True); open(f"{outdir}/metrics.jsonl","a").write(json.dumps(d)+"\n")
    df=pd.read_csv("/mnt/ecai/folds.csv")
    tr=df[df.fold!=a.fold].reset_index(drop=True); va=df[df.fold==a.fold].reset_index(drop=True)
    tr=tr[["filepath","label"]]
    if a.pseudo:
        pdf=pd.read_csv(a.pseudo)[["filepath","label"]]; tr=pd.concat([tr,pdf],ignore_index=True)
    dtr=DS(tr.filepath.values,tr.label.values.astype("float32"),build_aug(a.aug,a.imgsz,True))
    dva=DS(va.filepath.values,va.label.values.astype("float32"),build_aug(a.aug,a.imgsz,False))
    ltr=DataLoader(dtr,a.bs,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True,persistent_workers=True)
    lva=DataLoader(dva,a.bs*2,shuffle=False,num_workers=a.workers,pin_memory=True,persistent_workers=True)
    t_build=time.time()
    model=timm.create_model(a.model,pretrained=True,num_classes=1).to(dev).to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5); crit=nn.BCEWithLogitsLoss()
    steps=len(ltr)*a.epochs; warm=max(1,len(ltr))
    lr_at=lambda s: s/warm if s<warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm)))
    log({"start":a.exp,"model":a.model,"aug":a.aug,"imgsz":a.imgsz,"bs":a.bs,"lr":a.lr,"epochs":a.epochs,
         "ntr":len(tr),"nva":len(va),"params_M":round(sum(p.numel() for p in model.parameters())/1e6,1),
         "gpu":torch.cuda.get_device_name(0),"build_s":round(time.time()-t_build,1)})
    g=0
    for ep in range(a.epochs):
        model.train(); t0=time.time(); run=0.0
        for x,y in ltr:
            for pg in opt.param_groups: pg["lr"]=a.lr*lr_at(g)
            x=x.to(dev,non_blocking=True,memory_format=torch.channels_last); y=y.to(dev,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                loss=crit(model(x).squeeze(1),y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); run+=loss.item(); g+=1
        model.eval(); ps=[]; ys=[]; vl=0.0
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            for x,y in lva:
                x=x.to(dev,non_blocking=True,memory_format=torch.channels_last)
                o=model(x).squeeze(1).float(); vl+=crit(o,y.to(dev)).item()
                ps.append(torch.sigmoid(o).cpu().numpy()); ys.append(y.numpy())
        ps=np.concatenate(ps); ys=np.concatenate(ys)
        log({"epoch":ep,"train_loss":round(run/len(ltr),4),"val_loss":round(vl/len(lva),4),
             "val_auc":round(float(roc_auc_score(ys,ps)),4),
             "apcer_at_1pct_bpcer":round(apcer_at_bpcer(ys,ps),4),"sec":round(time.time()-t0,1)})
    np.save(f"{outdir}/oof.npy",np.stack([va.label.values,ps],1))
    torch.save({"model":model.state_dict(),"cfg":vars(a)},f"{outdir}/last.pt")
    # ---- inference on public_test with hflip TTA ----
    paths=sorted(glob.glob("/mnt/ecai/data/public_test/public_test/*.jpeg"))
    dte=DataLoader(DS(paths,None,build_aug(a.aug,a.imgsz,False)),a.bs*2,shuffle=False,num_workers=a.workers,pin_memory=True)
    ids=[]; sc=[]
    with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
        for x,bid in dte:
            x=x.to(dev,non_blocking=True,memory_format=torch.channels_last)
            o=torch.sigmoid(model(x).squeeze(1).float())
            o2=torch.sigmoid(model(torch.flip(x,dims=[3])).squeeze(1).float())
            sc.append(((o+o2)/2).cpu().numpy()); ids+=list(bid)
    pd.DataFrame({"id":ids,"label":np.concatenate(sc)}).to_csv(f"{outdir}/pred_public.csv",index=False)
    log({"done":True,"exp":a.exp,"n_pred":len(ids)})

if __name__=="__main__": main()
