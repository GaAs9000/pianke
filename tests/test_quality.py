import importlib
import os
import sys
import types
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageDraw

from pic_selecter.quality import analyze_image


def checkerboard(size=256, block=8):
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    for y in range(0, size, block):
        for x in range(0, size, block):
            if (x // block + y // block) % 2 == 0:
                draw.rectangle((x, y, x + block - 1, y + block - 1), fill="black")
    return img


def test_blurry_image_is_rejected_more_than_sharp_image():
    sharp = checkerboard()
    blurry = sharp.filter(ImageFilter.GaussianBlur(radius=8))

    sharp_q = analyze_image(sharp, file_size=200_000, strength="standard")
    blurry_q = analyze_image(blurry, file_size=200_000, strength="standard")

    assert sharp_q.blur_score > blurry_q.blur_score
    assert "blurry" not in sharp_q.flags
    assert {"blurry", "very_blurry"} & set(blurry_q.flags)
    assert blurry_q.auto_reject is True


def test_under_and_over_exposed_images_are_rejected():
    dark = Image.new("RGB", (256, 256), (4, 4, 4))
    bright = Image.new("RGB", (256, 256), (252, 252, 252))

    dark_q = analyze_image(dark, file_size=200_000, strength="standard")
    bright_q = analyze_image(bright, file_size=200_000, strength="standard")

    assert "underexposed" in dark_q.flags
    assert dark_q.auto_reject is True
    assert "overexposed" in bright_q.flags
    assert bright_q.auto_reject is True


def test_low_information_image_is_rejected():
    flat = Image.new("RGB", (256, 256), (128, 128, 128))

    q = analyze_image(flat, file_size=200_000, strength="standard")

    assert "low_information" in q.flags
    assert q.auto_reject is True
    assert q.reject_reason


def test_prescreen_strength_changes_blur_strictness():
    mildly_blurry = checkerboard(block=16).filter(ImageFilter.GaussianBlur(radius=2.2))

    conservative = analyze_image(mildly_blurry, file_size=200_000, strength="conservative")
    aggressive = analyze_image(mildly_blurry, file_size=200_000, strength="aggressive")

    assert aggressive.blur_score == conservative.blur_score
    assert aggressive.quality_score <= conservative.quality_score
    assert ("blurry" in aggressive.flags) or aggressive.auto_reject


def test_quality_info_can_be_serialized_to_plain_dict():
    q = analyze_image(checkerboard(), file_size=123_456, strength="standard")
    data = q.to_dict()

    assert data["blur_score"] == q.blur_score
    assert data["file_size"] == 123_456
    assert isinstance(data["flags"], list)


def test_scan_folder_pairs_raw_with_jpg_and_skips_root_outputs(tmp_path):
    sys.modules.setdefault("imagehash", types.SimpleNamespace(phash=lambda *args, **kwargs: "0" * 16))
    grouper = importlib.import_module("pic_selecter.grouper")
    (tmp_path / "IMG_0001.CR2").write_bytes(b"raw")
    (tmp_path / "IMG_0001.JPG").write_bytes(b"jpg")
    (tmp_path / "winners").mkdir()
    (tmp_path / "winners" / "old.jpg").write_bytes(b"old")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "winners").mkdir()
    (tmp_path / "nested" / "winners" / "kept.jpg").write_bytes(b"kept")

    pairs = grouper.scan_folder(str(tmp_path))

    simplified = {
        Path(primary).name: [Path(c).name for c in companions]
        for primary, companions in pairs
    }
    assert simplified["IMG_0001.CR2"] == ["IMG_0001.JPG"]
    assert "old.jpg" not in simplified
    assert simplified["kept.jpg"] == []


def test_compute_infos_reuses_fast_analysis_cache(tmp_path, monkeypatch):
    grouper = importlib.import_module("pic_selecter.grouper")
    img_path = tmp_path / "sample.jpg"
    checkerboard().save(img_path)
    calls = []

    def fake_process(path, strength="standard", face_aware=True, engine="expert",
                     llm_model=None, companions=None, cache_folder=None):
        calls.append(path)
        st = os.stat(path)
        return grouper.ImageInfo(
            path=path,
            phash="a" * 16,
            timestamp=None,
            size=st.st_size,
            mtime=st.st_mtime,
            exif_summary={"width": 256, "height": 256, "file_size": st.st_size},
            quality={"quality_score": 88.0, "flags": [], "auto_reject": False},
            companions=list(companions or []),
            dhash="b" * 16,
            whash="c" * 16,
            ahash="d" * 16,
            color_hist=np.array([1.0, 0.0], dtype=np.float32),
            orb_descs=np.zeros((8, 32), dtype=np.uint8),
            orb_kps=np.zeros((8, 2), dtype=np.float32),
        ), None

    monkeypatch.setattr(grouper, "_process_one", fake_process)

    first, skipped = grouper.compute_infos(str(tmp_path), engine="fast", workers=1)
    assert skipped == []
    assert len(first) == 1
    assert calls == [str(img_path)]

    calls.clear()
    second, skipped = grouper.compute_infos(str(tmp_path), engine="fast", workers=1)
    assert skipped == []
    assert calls == []
    assert second[0].phash == "a" * 16
    np.testing.assert_array_equal(second[0].orb_descs, np.zeros((8, 32), dtype=np.uint8))

    new_time = img_path.stat().st_mtime + 5
    os.utime(img_path, (new_time, new_time))
    calls.clear()
    third, skipped = grouper.compute_infos(str(tmp_path), engine="fast", workers=1)
    assert skipped == []
    assert len(third) == 1
    assert calls == [str(img_path)]
