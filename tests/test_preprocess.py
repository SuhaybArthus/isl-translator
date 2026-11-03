import os

import numpy as np
import pandas as pd

from src.data.dataset import ISLDataset, build_label_map, ROOT
from src.inference.preprocess import preprocess_clip

MP = os.path.join(ROOT, "data", "processed", "manifest_split.csv")


def test_matches_training_pipeline():
    lm = build_label_map(MP)
    for fill in (False, True):
        for use_z in (True, False):
            ds = ISLDataset(MP, "test", lm, seq_len=64, use_z=use_z, fill=fill)
            for i in range(0, len(ds), 7):
                x_train, _ = ds[i]
                raw = np.load(os.path.join(ROOT, ds.df.loc[i, "path"]))
                x_serve = preprocess_clip(raw, 64, use_z=use_z, fill=fill)
                np.testing.assert_allclose(x_train.numpy(), x_serve, atol=1e-6)


def test_rejects_bad_shape():
    import pytest
    with pytest.raises(ValueError):
        preprocess_clip(np.zeros((10, 100)))