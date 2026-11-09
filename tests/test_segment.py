import numpy as np

from src.inference.segment import SignSegmenter


def frame(raised):
    f = np.zeros(225, dtype=np.float32)
    f[11 * 3:11 * 3 + 3] = [0.4, 0.3, 0.0]      # shoulders, width 0.2 -> rest line at y=0.62
    f[12 * 3:12 * 3 + 3] = [0.6, 0.3, 0.0]
    f[15 * 3:15 * 3 + 3] = [0.4, 0.4 if raised else 0.8, 0.0]
    f[16 * 3:16 * 3 + 3] = [0.6, 0.8, 0.0]
    return f


def run(pattern, **kw):
    seg, out = SignSegmenter(**kw), []
    for r in pattern:
        s = seg.push(frame(r))
        if s is not None:
            out.append(s)
    s = seg.flush()
    if s is not None:
        out.append(s)
    return out


def test_one_sign():
    out = run([0] * 10 + [1] * 30 + [0] * 15)
    assert len(out) == 1 and 30 <= len(out[0]) <= 50


def test_two_signs():
    assert len(run([0] * 5 + [1] * 25 + [0] * 12 + [1] * 25 + [0] * 12)) == 2


def test_short_blip_ignored():
    assert run([0] * 10 + [1] * 2 + [0] * 15) == []


def test_flush_emits_unfinished_sign():
    assert len(run([0] * 5 + [1] * 30)) == 1


def test_segmenter_rest_state():
    seg = SignSegmenter(rest_line=1.6, pre_roll=12, post_pad=8)
    # Zero landmark vector representing rest
    rest_frame = np.zeros(225, dtype=np.float32)
    res = seg.push(rest_frame)
    assert res is None
    assert seg.active is False


def test_segmenter_reset():
    seg = SignSegmenter()
    seg.reset()
    assert seg.active is False