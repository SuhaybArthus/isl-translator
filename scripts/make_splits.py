import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "data", "processed", "manifest.csv")
SPLIT_DIR = os.path.join(ROOT, "data", "raw", "splits")
OUT = os.path.join(ROOT, "data", "processed", "manifest_split.csv")


def split_key(line):
    # "Animals/1. Dog/MVI_2978.MOV" -> ("animals", "1. dog", "mvi_2978")
    p = line.strip().replace("\\", "/").split("/")
    return (p[0].lower(), p[1].lower(), os.path.splitext(p[-1])[0].lower())


def manifest_key(row):
    # path: .../<Category_NofM>/<Category>/<NN. Sign>[/Extra]/MVI_x.npy
    p = row["path"].replace("\\", "/").split("/")
    i = p.index(row["category"])
    return (p[i].lower(), p[i + 1].lower(), os.path.splitext(p[-1])[0].lower())


lookup = {}
for name in ["train", "val", "test"]:
    with open(os.path.join(SPLIT_DIR, f"include_{name}.txt"), encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lookup[split_key(line)] = name
print("Official split entries:", len(lookup))

df = pd.read_csv(MANIFEST)
df = df[~df["video"].str.contains(r"\(\d+\)", regex=True)].copy()
df["split"] = df.apply(lambda r: lookup.get(manifest_key(r)), axis=1)

print("\nSplit counts:")
print(df["split"].value_counts(dropna=False))

unmatched = df[df["split"].isna()]
if len(unmatched):
    print(f"\nUnmatched clips ({len(unmatched)}):")
    print(unmatched[["category", "label", "video"]].to_string())

df = df.dropna(subset=["split"])
print("\nClips per label per split:")
print(pd.crosstab(df["label"], df["split"]))

df.to_csv(OUT, index=False)
print(f"\nSaved {OUT}")