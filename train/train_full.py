import os, sys, json, time, math, argparse, warnings, glob
warnings.simplefilter("ignore")
import numpy as np, pandas as pd, cv2, torch, torch.nn as nn
cv2.setNumThreads(0)
from torch.utils.data import DataLoader
import timm
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # find train_sweep.py alongside this file
from train_sweep import build_aug, DS

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--exp",required=True); ap.add_argument("--model",required=True)
    ap.add_argument("--imgsz",type=int,default=512); ap.add_argument("--bs",type=int,default=8)
    ap.add_argument("--lr",type=float,default=1e-4); ap.add_argument("--epochs",type=int,required=True)
    ap.add_argument("--save_epochs",required=True); ap.add_argument("--aug",default="v1")
    ap.add_argument("--workers",type=int,default=12); ap.add_argument("--smoke",action="store_true")
    ap.add_argument("--pseudo",default="")
    a=ap.parse_args()
    torch.backends.cudnn.benchmark=True; torch.backends.cuda.matmul.allow_tf32=True
    dev="cuda"; outdir=f"/mnt/ecai/exp/{a.exp}"; os.makedirs(outdir,exist_ok=True)
    def log(d): print(json.dumps(d),flush=True); open(f"{outdir}/metrics.jsonl","a").write(json.dumps(d)+"\n")
    save_set=set(int(x) for x in a.save_epochs.split(","))
    df=pd.read_csv("/mnt/ecai/folds.csv")[["filepath","label"]]  # FULL DATA: all folds, no validation
    if a.smoke: df=df.head(256)
    if a.pseudo:
        pdf=pd.read_csv(a.pseudo)[["filepath","label"]]; df=pd.concat([df,pdf],ignore_index=True)
    dtr=DS(df.filepath.values,df.label.values.astype("float32"),build_aug(a.aug,a.imgsz,True))
    ltr=DataLoader(dtr,a.bs,shuffle=True,num_workers=a.workers,pin_memory=True,drop_last=True,persistent_workers=True)
    t0=time.time()
    model=timm.create_model(a.model,pretrained=True,num_classes=1).to(dev).to(memory_format=torch.channels_last)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-5); crit=nn.BCEWithLogitsLoss()
    steps=len(ltr)*a.epochs; warm=max(1,len(ltr))
    lr_at=lambda s: s/warm if s<warm else 0.5*(1+math.cos(math.pi*(s-warm)/max(1,steps-warm)))
    log({"start":a.exp,"model":a.model,"aug":a.aug,"imgsz":a.imgsz,"bs":a.bs,"lr":a.lr,"epochs":a.epochs,
         "save_epochs":sorted(save_set),"n_train":len(df),"mode":"FULL_no_val",
         "params_M":round(sum(p.numel() for p in model.parameters())/1e6,1),"build_s":round(time.time()-t0,1)})
    g=0
    for ep in range(a.epochs):
        model.train(); te=time.time(); run=0.0
        for x,y in ltr:
            for pg in opt.param_groups: pg["lr"]=a.lr*lr_at(g)
            x=x.to(dev,non_blocking=True,memory_format=torch.channels_last); y=y.to(dev,non_blocking=True)
            with torch.autocast("cuda",dtype=torch.bfloat16): loss=crit(model(x).squeeze(1),y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); run+=loss.item(); g+=1
        log({"epoch":ep+1,"train_loss":round(run/len(ltr),4),"lr":round(opt.param_groups[0]["lr"],6),"sec":round(time.time()-te,1)})
        if (ep+1) in save_set:
            torch.save({"model":model.state_dict(),"cfg":vars(a),"epoch":ep+1},f"{outdir}/ckpt_e{ep+1}.pt")
            log({"saved_ckpt":ep+1})
    # ---- inference public_test (hflip TTA) for each saved checkpoint ----
    paths=sorted(glob.glob("/mnt/ecai/data/public_test/public_test/*.jpeg"))
    dte=DataLoader(DS(paths,None,build_aug(a.aug,a.imgsz,False)),a.bs*2,shuffle=False,num_workers=a.workers,pin_memory=True)
    for N in sorted(save_set):
        cp=f"{outdir}/ckpt_e{N}.pt"
        if not os.path.exists(cp): continue
        model.load_state_dict(torch.load(cp,map_location=dev)["model"]); model.eval()
        ids=[]; sc=[]
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            for x,bid in dte:
                x=x.to(dev,non_blocking=True,memory_format=torch.channels_last)
                o=torch.sigmoid(model(x).squeeze(1).float())
                o2=torch.sigmoid(model(torch.flip(x,dims=[3])).squeeze(1).float())
                sc.append(((o+o2)/2).cpu().numpy()); ids+=list(bid)
        pd.DataFrame({"id":ids,"label":np.concatenate(sc)}).to_csv(f"{outdir}/pred_e{N}.csv",index=False)
        log({"infer_ckpt":N,"n_pred":len(ids)})
    log({"done":True,"exp":a.exp})

if __name__=="__main__": main()
