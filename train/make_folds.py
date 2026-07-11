import pandas as pd, os
from sklearn.model_selection import StratifiedKFold
ROOT="/mnt/ecai/data"
df=pd.read_csv(f"{ROOT}/train_labels.csv")
df["filepath"]=df["id"].map(lambda x: f"{ROOT}/train/train/{x}.jpeg")
assert df["filepath"].map(os.path.exists).all(), "missing files!"
df["strat"]=df["label"].astype(str)+"_"+df["type"].astype(str)
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
df["fold"]=-1
for f,(_,vi) in enumerate(skf.split(df,df["strat"])):
    df.loc[vi,"fold"]=f
out=df[["id","filepath","label","is_digital","type","fold"]]
out.to_csv("/mnt/ecai/folds.csv",index=False)
print("saved /mnt/ecai/folds.csv  rows",len(out))
print("\nfold sizes:\n",out["fold"].value_counts().sort_index())
print("\nfold x label:\n",pd.crosstab(out["fold"],out["label"]))
print("\nfold0 val: label\n",out[out.fold==0]["label"].value_counts().to_dict())
print("fold0 val: type\n",out[out.fold==0]["type"].value_counts().to_dict())
