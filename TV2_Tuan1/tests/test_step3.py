import numpy as np
import pytest

import src.step3 as step3
from src.step3 import synthesize_three_channels


def test_channel_order_and_subtraction_formula(monkeypatch):
    monkeypatch.setattr(step3, "compute_3d_clahe", lambda volume_zyx, **_: volume_zyx)
    pre = np.arange(64, dtype=np.float32).reshape(4, 4, 4) + 10
    post = pre + np.linspace(0, 20, 64, dtype=np.float32).reshape(4, 4, 4)
    image = synthesize_three_channels(
        pre, post, apply_denoise=False, clip_percentiles=(0, 100),
        clahe_kernel_size=2, output_dtype="float32",
    )
    expected_c1 = (post - post.mean()) / (post.std() + 1e-8)
    sub = np.maximum(0, post - pre)
    expected_c2 = (sub - sub.mean()) / (sub.std() + 1e-8)
    assert image.shape == (3, 4, 4, 4)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[0], expected_c1, atol=1e-5)
    np.testing.assert_allclose(image[1], expected_c2, atol=1e-5)
    assert np.isfinite(image).all()


def test_constant_and_zero_pre_are_finite():
    pre = np.zeros((4, 4, 4), dtype=np.float32)
    post = np.ones_like(pre)
    image = synthesize_three_channels(
        pre, post, apply_denoise=False, clahe_kernel_size=2,
    )
    assert image.dtype == np.float16
    assert np.isfinite(image).all()
    assert np.count_nonzero(image) == 0


def test_mismatched_voi_shape_is_rejected():
    with pytest.raises(ValueError, match="shape"):
        synthesize_three_channels(
            np.ones((4, 4, 4), dtype=np.float32),
            np.ones((4, 4, 5), dtype=np.float32),
            apply_denoise=False,
        )


def test_actual_clahe_on_3d_volume_when_scipy_is_available():
    pytest.importorskip("scipy.linalg")
    volume = np.linspace(0, 1, 8 ** 3, dtype=np.float32).reshape(8, 8, 8)
    result = step3.compute_3d_clahe(volume, kernel_size=2)
    assert result.shape == volume.shape
    assert np.isfinite(result).all()
