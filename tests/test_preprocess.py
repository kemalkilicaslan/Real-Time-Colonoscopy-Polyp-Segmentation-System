import cv2
import numpy as np
import pytest

from polypseg.preprocess import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    eval_transforms,
    normalize,
    postprocess,
    preprocess,
    read_image,
    read_mask,
    train_transforms,
)


def test_normalize_layout_and_values():
    rgb = np.full((4, 6, 3), 255, np.uint8)
    x = normalize(rgb)
    assert x.shape == (3, 4, 6) and x.dtype == np.float32
    np.testing.assert_allclose(x[:, 0, 0], (1 - IMAGENET_MEAN) / IMAGENET_STD, rtol=1e-6)


def test_preprocess_shape():
    rgb = np.random.randint(0, 255, (288, 384, 3), np.uint8)
    assert preprocess(rgb, 352).shape == (1, 3, 352, 352)


def test_postprocess_restores_size_and_probability_range():
    logits = np.random.randn(1, 1, 352, 352).astype(np.float32) * 10
    prob = postprocess(logits, (288, 384))
    assert prob.shape == (288, 384)
    assert prob.min() >= 0 and prob.max() <= 1


def test_read_mask_binarises_jpeg_like_values(tmp_path):
    m = np.zeros((10, 10), np.uint8)
    m[:5] = 250  # JPEG artefacts around 255
    m[5:, :2] = 3  # and around 0
    p = tmp_path / "m.png"
    cv2.imwrite(str(p), m)
    out = read_mask(p)
    assert set(np.unique(out)) == {0, 1}
    assert out.sum() == 50


def test_read_image_is_rgb(tmp_path):
    bgr = np.zeros((4, 4, 3), np.uint8)
    bgr[..., 2] = 200  # red in BGR
    p = tmp_path / "i.png"
    cv2.imwrite(str(p), bgr)
    assert read_image(p)[0, 0, 0] == 200


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "nope.png")


@pytest.mark.parametrize("tf", [train_transforms(64), eval_transforms(64)])
def test_transforms_keep_mask_binary_and_aligned(tf):
    img = np.random.randint(0, 255, (100, 120, 3), np.uint8)
    mask = np.zeros((100, 120), np.uint8)
    mask[30:70, 40:80] = 1
    out = tf(image=img, mask=mask)
    assert out["image"].shape == (64, 64, 3)
    assert out["mask"].shape == (64, 64)
    assert set(np.unique(out["mask"])) <= {0, 1}


def test_read_image_keeps_colour_of_minisblack_tiff(tmp_path):
    """Regression: CVC-ClinicDB TIFFs are RGB tagged as MINISBLACK; OpenCV reads them as grey."""
    import tifffile

    rgb = np.zeros((8, 8, 3), np.uint8)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 200, 100, 50
    p = tmp_path / "frame.tif"
    tifffile.imwrite(p, rgb, photometric="minisblack", planarconfig="contig")
    assert cv2.imread(str(p), cv2.IMREAD_COLOR)[0, 0].tolist() == [200, 200, 200]  # the trap
    assert read_image(p)[0, 0].tolist() == [200, 100, 50]


def test_read_mask_tiff(tmp_path):
    import tifffile

    m = np.zeros((8, 8), np.uint8)
    m[2:5, 2:5] = 255
    p = tmp_path / "mask.tif"
    tifffile.imwrite(p, m)
    assert read_mask(p).sum() == 9
