"""CPU-only tests for the Mask2Former wrapper (tiny random model, no downloads)."""

import numpy as np
import pytest
import torch
from PIL import Image

from iiss.models import m2f
from iiss.utils.config import build_config

TINY = {
    "backbone_config": {
        "model_type": "swin",
        "embed_dim": 16,
        "depths": [1, 1, 1, 1],
        "num_heads": [1, 1, 2, 2],
        "window_size": 4,
        "out_features": ["stage1", "stage2", "stage3", "stage4"],
    },
    "feature_size": 32,
    "mask_feature_size": 32,
    "hidden_dim": 32,
    "encoder_feedforward_dim": 64,
    "dim_feedforward": 64,
    "encoder_layers": 1,
    "decoder_layers": 2,
    "num_attention_heads": 2,
    "num_queries": 6,
    "train_num_points": 64,
}


@pytest.fixture
def cfg() -> dict:
    cfg = build_config("m2f", {"input_size": 64})
    cfg["amp"] = False
    cfg["model_config"] = TINY
    return cfg


def test_pad_batch_normalises_and_pads_to_32():
    imgs = [torch.ones(3, 40, 64), torch.zeros(3, 64, 30)]
    masks = [torch.ones(2, 40, 64, dtype=torch.uint8), torch.zeros(0, 64, 30, dtype=torch.uint8)]
    pixels, pixel_mask, padded = m2f.pad_batch(imgs, masks)
    assert pixels.shape == (2, 3, 64, 64)
    assert pixel_mask[0, :40].all() and not pixel_mask[0, 40:].any()
    assert pixel_mask[1, :, :30].all() and not pixel_mask[1, :, 30:].any()
    assert torch.allclose(pixels[0, 0, 0, 0], torch.tensor((1 - 0.485) / 0.229))
    assert pixels[0, :, 40:].abs().sum() == 0  # padding stays zero after normalisation
    assert padded[0].shape == (2, 64, 64) and padded[0][:, 40:].sum() == 0
    assert padded[1].shape == (0, 64, 64)


def test_poly_factor():
    tcfg = {"poly_power": 0.9, "warmup_iters": 0}
    assert m2f.poly_factor(0, 100, tcfg) == pytest.approx(1.0)
    assert m2f.poly_factor(50, 100, tcfg) == pytest.approx(0.5**0.9)
    assert m2f.poly_factor(100, 100, tcfg) == 0.0
    warm = {**tcfg, "warmup_iters": 10, "warmup_factor": 0.1}
    assert m2f.poly_factor(0, 100, warm) == pytest.approx(0.1)


def test_param_groups_backbone_lr_and_no_decay(cfg: dict):
    model = m2f.build_model(cfg, pretrained=False)
    tcfg = {**cfg["train"], "lr": 1e-4}
    groups = m2f.param_groups(model, tcfg)
    n_params = sum(len(g["params"]) for g in groups)
    assert n_params == sum(1 for p in model.parameters() if p.requires_grad)
    lrs = sorted({round(g["lr"], 10) for g in groups})
    assert lrs == pytest.approx([1e-5, 1e-4])
    by_id = {id(p): g for g in groups for p in g["params"]}
    named = dict(model.named_parameters())
    assert by_id[id(named["model.transformer_module.queries_embedder.weight"])]["weight_decay"] == 0
    weight = next(p for n, p in named.items() if n.startswith(m2f.BACKBONE_PREFIX) and p.ndim > 1)
    assert by_id[id(weight)]["weight_decay"] == pytest.approx(0.05)


def test_instances_from_logits_crops_ignores_class0_and_ranks():
    q, c = 3, 4
    class_logits = torch.full((q, c + 1), -10.0)
    class_logits[0, 0] = 10.0  # class 0 only -> must never be returned as class 0
    class_logits[1, 2] = 10.0
    class_logits[2] = 0.0
    class_logits[2, 3] = 1.0  # p = e / (e + 4) ~ 0.40
    mask_logits = torch.full((q, 4, 4), -10.0)
    mask_logits[1, :2, :2] = 10.0  # top-left of the padded input
    mask_logits[2] = 10.0
    probs, labels, scores = m2f.instances_from_logits(
        class_logits, mask_logits, (16, 16), (8, 12), top_k=5
    )
    assert probs.shape == (5, 8, 12)
    assert 0 not in labels.tolist()
    assert labels[0].item() == 2 and torch.all(scores[:-1] >= scores[1:])
    assert probs[0][:8, :8].mean() > 0.9  # query 1 covers the top-left 8x8 of the padded input


def test_tiny_model_loss_and_predict_on_cpu(cfg: dict):
    torch.manual_seed(0)
    model = m2f.build_model(cfg, pretrained=False)
    images = [torch.rand(3, 48, 64), torch.rand(3, 64, 40)]
    masks = torch.zeros(2, 48, 64, dtype=torch.uint8)
    masks[0, :10, :10] = 1
    masks[1, 20:, 30:] = 1
    targets = [
        {"masks": masks, "labels": torch.tensor([3, 7])},
        {"masks": torch.zeros(0, 64, 40, dtype=torch.uint8), "labels": torch.zeros(0).long()},
    ]
    model.train()
    loss = m2f.loss_fn(model, images, targets)["loss"]
    assert torch.isfinite(loss)
    loss.backward()

    model.eval()
    timings: list[float] = []
    dets = m2f.predict_image(
        model, Image.new("RGB", (80, 40)), {"id": 9, "width": 80, "height": 40}, cfg, timings
    )
    assert len(timings) == 1 and len(dets) == cfg["eval"]["max_dets"]
    for det in dets:
        assert det.image_id == 9 and 1 <= det.category_id < 75
        assert det.mask.shape == (40, 80) and det.mask.dtype == np.bool_
