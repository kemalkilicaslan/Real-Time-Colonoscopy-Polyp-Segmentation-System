import numpy as np
import pytest
import torch

from polypseg.export import check_onnx, export_onnx
from polypseg.inference import OnnxSegmenter, TorchSegmenter
from polypseg.model import build_model
from polypseg.train import DiceBCELoss, batch_dice


def tiny_model():
    torch.manual_seed(0)
    return build_model("Unet", "resnet18", encoder_weights=None).eval()


def test_model_output_shape():
    with torch.no_grad():
        assert tiny_model()(torch.randn(2, 3, 64, 64)).shape == (2, 1, 64, 64)


def test_onnx_matches_torch(tmp_path):
    model = tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", image_size=64)
    assert check_onnx(model, path, 64, n=2) < 1e-3


def test_torch_and_onnx_segmenters_agree(tmp_path):
    model = tiny_model()
    path = export_onnx(model, tmp_path / "m.onnx", image_size=64)
    rgb = np.random.randint(0, 255, (90, 120, 3), np.uint8)
    a = TorchSegmenter(model, 64, torch.device("cpu"))(rgb)
    b = OnnxSegmenter(path)(rgb)
    assert a.shape == b.shape == (90, 120)
    assert np.abs(a - b).max() < 1e-3


def test_loss_is_low_for_perfect_logits_and_dice_is_one():
    y = torch.zeros(2, 1, 16, 16)
    y[:, :, 4:12, 4:12] = 1
    logits = (y * 2 - 1) * 20
    assert DiceBCELoss()(logits, y).item() < 0.01
    assert torch.allclose(batch_dice(logits, y), torch.ones(2))


def test_dice_term_ignores_empty_masks():
    """Regression: soft Dice on an empty mask is ~1 for a few uncertain pixels."""
    y = torch.zeros(1, 1, 16, 16)
    logits = torch.full((1, 1, 16, 16), -20.0)
    logits[..., 0, :4] = 0.0  # four uncertain pixels on a polyp-free frame
    bce_only = torch.nn.BCEWithLogitsLoss()(logits, y).item()
    assert abs(DiceBCELoss()(logits, y).item() - bce_only) < 1e-6


def test_epoch_scores_balances_positives_and_negatives():
    from polypseg.train import epoch_scores

    dice = torch.tensor([0.8, 0.6, 1.0, 0.0])
    frac = torch.tensor([0.1, 0.1, 0.0, 0.05])
    is_pos = torch.tensor([True, True, False, False])
    s = epoch_scores(dice, frac, is_pos)
    assert s["pos_dice"] == pytest.approx(0.7)
    assert s["neg_clean"] == pytest.approx(0.5)
    assert s["score"] == pytest.approx(0.6)
    only_pos = epoch_scores(dice[:2], frac[:2], is_pos[:2])
    assert only_pos["score"] == pytest.approx(0.7)
