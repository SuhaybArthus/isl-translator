"""Data-free tests for the shared preprocessing, so CI can run them without the dataset."""
import numpy as np
import pytest

from src.inference.preprocess import fill_gaps, preprocess_clip


def clip(T=50):
    a = np.random.default_rng(0).uniform(0.2, 0.8, (T, 225)).astype(np.float32)
    a[:, 33:36] = [0.4, 0.3, 0.0]               # landmark 11, left shoulder
    a[:, 36:39] = [0.6, 0.3, 0.0]               # landmark 12, right shoulder
    return a


def test_output_shape_is_fixed_and_finite():
    x = preprocess_clip(clip(30), seq_len=64)
    assert x.shape == (64, 225) and x.dtype == np.float32 and np.isfinite(x).all()


def test_without_z_drops_a_third_of_the_features():
    assert preprocess_clip(clip(), 64, use_z=False).shape == (64, 150)


def test_shoulder_midpoint_becomes_the_origin():
    x = preprocess_clip(clip(), 64, fill=False).reshape(64, 75, 3)
    assert np.allclose((x[:, 11] + x[:, 12]) / 2, 0, atol=1e-5)


def test_undetected_hand_stays_zero():
    a = clip()
    a[:, 99:162] = 0                              # left hand never detected
    x = preprocess_clip(a, 64).reshape(64, 75, 3)
    assert np.all(x[:, 33:54] == 0)


def test_short_gap_is_filled_and_long_gap_is_not():
    x = np.ones((30, 75, 3), dtype=np.float32)
    missing = np.zeros((30, 75), dtype=bool)
    missing[10:13, 33:54] = True                  # 3 frames missing: filled
    missing[20:28, 54:75] = True                  # 8 frames missing: left alone
    x[missing] = 0
    x2, m2 = fill_gaps(x, missing, max_gap=5)
    assert not m2[10:13, 33:54].any() and np.allclose(x2[10:13, 33:54], 1)
    assert m2[20:28, 54:75].all()


def test_bad_shapes_are_rejected():
    with pytest.raises(ValueError):
        preprocess_clip(np.zeros((10, 100)))
    with pytest.raises(ValueError):
        preprocess_clip(np.zeros((1, 225)))