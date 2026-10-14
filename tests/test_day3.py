import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest
from rasterio.transform import from_origin

from src.features import CHANNELS, build_features
from src.grid import Grid, read_raster, write_raster
from src.train_data import INVALID, event_id, iter_tar_samples, parse_sample, save_cache, to_db_auto
from tests.synth import RES, make_scene

REPO = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ data loading (no torch needed)
def _npy(a):
    b = io.BytesIO()
    np.save(b, a)
    return b.getvalue()


def make_tar(samples) -> io.BytesIO:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for key, fields in samples:
            for name, val in fields.items():
                data = json.dumps(val).encode() if name == "info.json" else _npy(val)
                ti = tarfile.TarInfo(f"{key}.{name}")
                ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
    buf.seek(0)
    return buf


def kuro_sample(linear=True, seed=0):
    rng = np.random.default_rng(seed)
    f = {}
    for k in ("flood_vv", "flood_vh", "sec1_vv", "sec1_vh", "sec2_vv", "sec2_vh"):
        db = rng.normal(-10, 2, (1, 32, 32)).astype(np.float32)
        f[f"{k}.npy"] = (10 ** (db / 10)) if linear else db
    mask = np.zeros((32, 32), np.uint8)
    mask[:8] = 2          # flood
    mask[8:12] = 1        # permanent water
    mask[12:14] = 3       # invalid class (as in the GeoBench version)
    valid = np.ones((32, 32), np.uint8)
    valid[30:] = 0
    f["mask.npy"], f["valid_mask.npy"] = mask, valid
    f["dem.npy"] = np.zeros((32, 32), np.float32)
    f["info.json"] = {"actid": 123, "aoiid": 7}
    return f


@pytest.mark.parametrize("linear", [True, False])
def test_tar_stream_parse_and_unit_detection(linear):
    tar = make_tar([("s0", kuro_sample(linear, 0)), ("s1", kuro_sample(linear, 1))])
    samples = list(iter_tar_samples(tar))
    assert len(samples) == 2 and "flood_vv" in samples[0] and samples[0]["info"]["actid"] == 123
    X, y, ev = parse_sample(samples[0])
    assert X.shape == (6, 32, 32) and abs(np.nanmedian(X) - (-10)) < 1.0      # linear and dB agree after detection
    assert (y[:8] == 1).all() and (y[8:12] == 0).all()                         # flood=1, permanent water=0
    assert (y[12:14] == INVALID).all() and (y[30:] == INVALID).all()           # invalid class / valid_mask=0
    assert ev == "123_7"


def test_unit_detection_and_event_fallback():
    assert to_db_auto(np.array([0.1, 0.01]))[0] == pytest.approx(-10.0)
    assert to_db_auto(np.array([-10.0, -20.0]))[0] == -10.0
    assert event_id(None) == "unknown" and event_id({"x": 1}) == "unknown"
    assert parse_sample({"flood_vv": np.zeros((4, 4))}) is None


def test_features_shape_scale_and_nan():
    a = np.full((8, 8), -12.0, np.float32)
    f = build_features(a, a - 7, a + 3, a - 4)
    assert f.shape == (len(CHANNELS), 8, 8) and abs(f[0, 0, 0] - 0.3) < 1e-6
    assert f[4, 0, 0] == pytest.approx(-0.6)                                   # (-12 - -9)/5
    assert not np.isnan(build_features(np.full((2, 2), np.nan), a[:2, :2], a[:2, :2], a[:2, :2])).any()


torch = pytest.importorskip("torch")
pytest.importorskip("segmentation_models_pytorch")

from src import evaluate, infer_flood, train  # noqa: E402
from src.model import Confusion, bce_dice_loss, load_checkpoint  # noqa: E402


def test_loss_ignores_invalid_pixels_and_prefers_correct_logits():
    t = torch.zeros(1, 4, 4, dtype=torch.long)
    t[0, :2] = 1
    t[0, 3] = 255
    good = torch.where(t == 1, 8.0, -8.0).float()[:, None]
    bad = -good
    bad[:, :, 3] = 50.0                      # garbage on ignored pixels must not matter for `good`
    good2 = good.clone(); good2[:, :, 3] = 50.0
    assert bce_dice_loss(good, t) == pytest.approx(bce_dice_loss(good2, t), abs=1e-6)
    assert bce_dice_loss(good, t) < 0.05 < bce_dice_loss(bad, t)


def test_confusion_metrics():
    c = Confusion()
    pred = np.array([1, 1, 0, 0, 1], bool)
    tgt = np.array([1, 0, 1, 0, 255])
    c.update(pred, tgt)
    m = c.metrics()
    assert (c.tp, c.fp, c.fn, c.tn) == (1, 1, 1, 1)
    assert m["iou"] == pytest.approx(1 / 3) and m["f1"] == pytest.approx(0.5)


def test_event_holdout_split_has_no_leakage():
    ev = np.array([f"e{i % 8}" for i in range(80)])
    tr, va, how = train.split_by_event(ev, 0.25)
    assert how == "event" and not (set(ev[tr]) & set(ev[va])) and len(va) > 0
    _, _, how2 = train.split_by_event(np.array(["unknown"] * 20))
    assert how2 == "random"


def synthetic_cache(tmp_path: Path, n_scenes=14, size=64):
    X, Y, E = [], [], []
    rng = np.random.default_rng(0)
    for sd in range(n_scenes):
        s = make_scene(seed=sd)
        pre, post, tr = s["s1_pre"], s["s1_post"], s["truth"]["flood_new"]
        for col in (0, 40, 110, 170, 186, 230):
            r0 = 118
            sl = (slice(r0, r0 + size), slice(col, col + size))
            sec2 = pre[:, sl[0], sl[1]] + rng.normal(0, 0.5, pre[:, sl[0], sl[1]].shape)
            X.append(np.concatenate([post[:, sl[0], sl[1]], pre[:, sl[0], sl[1]], sec2]).astype(np.float16))
            Y.append(tr[sl].astype(np.uint8))
            E.append(f"ev{sd % 7}")
    save_cache(tmp_path, "train", X, Y, E)
    save_cache(tmp_path, "test", X[:30], Y[:30], E[:30])
    return tmp_path


def test_end_to_end_train_evaluate_infer_on_synthetic_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cache = synthetic_cache(tmp_path / "cache")
    ck = tmp_path / "models" / "m.pt"
    train.main(["--cache", str(cache), "--out", str(ck), "--encoder", "resnet18", "--no-pretrained",
                "--epochs", "6", "--batch-size", "8", "--workers", "0", "--lr", "2e-3"])
    model, meta = load_checkpoint(ck)
    assert meta["val"]["f1"] > 0.6, meta["val"]          # trivially separable task: must learn it on CPU in seconds

    evaluate.main(["--cache", str(cache), "--weights", str(ck), "--max-test", "30"])
    rep = json.loads((tmp_path / "reports" / "eval_metrics.json").read_text(encoding="utf-8"))
    assert rep["model"]["f1"] > 0.6 and 0.0 <= rep["baseline"]["f1"] <= 1.0
    assert "Known failure modes" in (tmp_path / "reports" / "ai_component.md").read_text(encoding="utf-8")

    # tuning writes a usable thresholds file
    evaluate.main(["--cache", str(cache), "--tune"])
    thr = json.loads((tmp_path / "models" / "baseline_thresholds.json").read_text(encoding="utf-8"))
    assert -22 <= thr["vv_water_db"] <= -12

    # full-scene inference through the real raster path
    s = make_scene(seed=99)
    grid = Grid("EPSG:32645", from_origin(500000.0, 3100000.0, RES, RES), 300, 300, RES)
    proc = tmp_path / "proc"
    write_raster(proc / "s1_pre.tif", s["s1_pre"], grid, nodata=np.nan)
    write_raster(proc / "s1_post.tif", s["s1_post"], grid, nodata=np.nan)
    write_raster(proc / "dem.tif", s["dem"], grid, nodata=np.nan)
    infer_flood.main(["--config", str(REPO / "config" / "aoi_trishuli.yaml"), "--weights", str(ck),
                      "--proc", str(proc), "--out", str(tmp_path / "out"), "--no-s2"])
    mask = read_raster(tmp_path / "out" / "flood_mask_ml.tif")[0][0].astype(bool)
    truth = s["truth"]["flood_new"]
    f1 = 2 * (mask & truth).sum() / (mask.sum() + truth.sum())
    assert f1 > 0.6 and (mask & s["truth"]["shadow"]).sum() == 0
    assert 0 <= json.loads((tmp_path / "out" / "ml_report.json").read_text(encoding="utf-8"))["flood_area_km2"] < 9.0
