import numpy as np

from src.inference.handedness import hand_share, mirrored, wrist_motion, wrist_travel


def test_mirroring_twice_gives_the_original():
    x = np.random.default_rng(1).normal(size=(64, 225)).astype(np.float32)
    assert np.allclose(mirrored(mirrored(x, True), True), x)


def test_mirroring_moves_the_left_hand_to_the_right_hand_and_flips_x():
    x = np.zeros((4, 225), dtype=np.float32).reshape(4, 75, 3)
    x[:, 33:54] = [0.5, 0.2, 0.1]                      # left hand only
    m = mirrored(x.reshape(4, 225), True).reshape(4, 75, 3)
    assert np.allclose(m[:, 54:75], [-0.5, 0.2, 0.1]) and np.all(m[:, 33:54] == 0)


def test_without_z_has_two_values_per_landmark():
    assert mirrored(np.ones((8, 150), dtype=np.float32), False).shape == (8, 150)


def test_wrist_travel_tells_the_moving_hand():
    a = np.zeros((10, 75, 3), dtype=np.float32)
    a[:, 15, 0] = np.linspace(0.1, 1.0, 10)            # left wrist moves
    a[:, 15, 1] = 0.5
    a[:, 16, :2] = 0.5                                 # right wrist still
    left, right = wrist_travel(a.reshape(10, 225), True)
    assert left > 0.5 and right == 0.0


def _segment(wrist_y):
    s = np.zeros((len(wrist_y), 225), dtype=np.float32)
    s[:, 33:35] = [0.4, 0.3]                           # shoulders 0.2 apart
    s[:, 36:38] = [0.6, 0.3]
    s[:, 45] = 0.4                                     # left wrist x
    s[:, 46] = wrist_y
    return s


def test_still_hands_count_as_no_motion():
    assert wrist_motion(_segment(np.full(20, 0.8))) == 0.0


def test_a_moving_wrist_counts_in_shoulder_widths():
    assert wrist_motion(_segment(np.linspace(0.4, 0.8, 20))) > 1.5     # 0.4 of travel / 0.2 width = 2.0


def test_hand_share_counts_frames_with_a_tracked_hand():
    s = np.zeros((10, 225), dtype=np.float32)
    s[:4, 100] = 0.5
    assert abs(hand_share(s) - 0.4) < 1e-6