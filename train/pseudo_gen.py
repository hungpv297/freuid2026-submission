import glob, numpy as np, pandas as pd, os
# strong, diverse maxvit predictions to ensemble for pseudo-labels
SRC=[
 "/mnt/ecai/exp/full_maxvit_b512_b12/pred_e12.csv",
 "/mnt/ecai/exp/full_maxvit_b512_b12/pred_e11.csv",
 "/mnt/ecai/exp/full_maxvit_b512_b12/pred_e10.csv",
 "/mnt/ecai/exp/ep08_maxvit_l512/pred_public.csv",
 "/mnt/ecai/exp/07_maxvit_l512_e15/pred_public.csv",
 "/mnt/ecai/exp/03_maxvit_b512_e15/pred_public.csv",
 "/mnt/ecai/exp/23_maxvit_b384_e15/pred_public.csv",
]
SRC=[s for s in SRC if os.path.exists(s)]
print("ensembling",len(SRC),"pred files for pseudo")
dfs=[pd.read_csv(s).set_index("id")["label"].rename(f"p{i}") for i,s in enumerate(SRC)]
M=pd.concat(dfs,axis=1); score=M.mean(axis=1)
LO,HI=0.05,0.95
df=pd.DataFrame({"id":score.index,"score":score.values})
df["filepath"]=df["id"].map(lambda x:f"/mnt/ecai/data/public_test/public_test/{x}.jpeg")
conf=df[(df.score<LO)|(df.score>HI)].copy()
conf["label"]=(conf.score>0.5).astype(int)
conf[["id","filepath","label","score"]].to_csv("/mnt/ecai/pseudo.csv",index=False)
print(f"total public_test={len(df)}  confident(<{LO} or >{HI})={len(conf)}  dropped={len(df)-len(conf)}")
print("pseudo label balance:",conf.label.value_counts().to_dict())
print("saved /mnt/ecai/pseudo.csv")
