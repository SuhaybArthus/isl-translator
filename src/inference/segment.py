"""Streaming sign segmentation from raw landmark frames.

A frame is a raw (225,) vector: pose(33)+left hand(21)+right hand(21), (x,y,z) each,
with y growing downward (MediaPipe image coordinates). A sign is a stretch of frames
where at least one wrist is above a rest line placed below the shoulders.
Output segments are raw (T,225), ready for preprocess_clip().
"""
from collections import deque

import numpy as np

L_SH, R_SH, L_WR, R_WR = 11, 12, 15, 16


def _pt(frame, i):
    return frame[i * 3:i * 3 + 3]


def hands_raised(frame, rest_line=1.2):
    """True if a wrist is above the rest line: shoulder-midpoint y + rest_line * shoulder width."""
    ls, rs = _pt(frame, L_SH), _pt(frame, R_SH)
    if not (ls.any() and rs.any()):
        return False                                   # pose not detected
    sw = np.linalg.norm(ls[:2] - rs[:2])
    if sw < 1e-6:
        return False
    line = (ls[1] + rs[1]) / 2 + rest_line * sw
    for w in (L_WR, R_WR):
        wr = _pt(frame, w)
        if wr.any() and wr[1] < line:
            return True
    return False


class SignSegmenter:
    def __init__(self, rest_line=1.6, start_frames=3, end_frames=8,
                 pre_roll=12, post_pad=8, min_len=12, max_len=160):
        self.rest_line, self.start_frames, self.end_frames = rest_line, start_frames, end_frames
        self.post_pad, self.min_len, self.max_len = post_pad, min_len, max_len
        self.pre = deque(maxlen=pre_roll + start_frames)
        self.reset()

    def reset(self):
        self.pre.clear()
        self.seg, self.active, self.up, self.down = [], False, 0, 0

    def push(self, frame):
        """Feed one raw frame. Returns a (T,225) segment when a sign just ended, else None."""
        frame = np.asarray(frame, dtype=np.float32)
        raised = hands_raised(frame, self.rest_line)
        if not self.active:
            self.pre.append(frame)
            self.up = self.up + 1 if raised else 0
            if self.up >= self.start_frames:
                self.active, self.seg, self.down = True, list(self.pre), 0
            return None
        self.seg.append(frame)
        self.down = 0 if raised else self.down + 1
        if self.down >= self.end_frames:
            return self._finish(trim=max(0, self.down - self.post_pad))
        if len(self.seg) >= self.max_len:
            return self._finish(trim=0)
        return None

    def flush(self):
        """Call at end of stream to emit a sign that was still in progress."""
        return self._finish(trim=max(0, self.down - self.post_pad)) if self.active else None

    def _finish(self, trim):
        seg = self.seg[:len(self.seg) - trim] if trim else self.seg
        self.reset()
        return np.stack(seg) if len(seg) >= self.min_len else None