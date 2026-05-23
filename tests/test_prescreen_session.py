import importlib
import sys
import time
import types
from pathlib import Path


def import_app_module():
    sys.modules.setdefault("imagehash", types.SimpleNamespace(phash=lambda *args, **kwargs: "0" * 16))
    if "flask" not in sys.modules:
        class FakeFlask:
            def __init__(self, *args, **kwargs):
                self.static_folder = kwargs.get("static_folder", "static")

            def route(self, *args, **kwargs):
                return lambda fn: fn

            def after_request(self, fn):
                return fn

            def before_request(self, fn):
                return fn

            def run(self, *args, **kwargs):
                return None

        sys.modules["flask"] = types.SimpleNamespace(
            Flask=FakeFlask,
            Response=lambda *args, **kwargs: types.SimpleNamespace(headers={}, *args, **kwargs),
            abort=lambda *args, **kwargs: None,
            jsonify=lambda *args, **kwargs: args[0] if args else kwargs,
            request=types.SimpleNamespace(path="/", host="localhost:5057", headers={}, method="GET", args={}),
            send_file=lambda *args, **kwargs: None,
            send_from_directory=lambda *args, **kwargs: None,
        )
    sys.modules.pop("app", None)
    return importlib.import_module("app")


def make_info(path: Path, score=80.0, auto_reject=False, reason=None, flags=None):
    app = import_app_module()
    path.write_bytes(b"fake")
    if flags is None:
        flags = ["very_blurry"] if auto_reject else []
    return app.ImageInfo(
        path=str(path),
        phash="0" * 16,
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
        exif_summary={"width": 1000, "height": 800, "file_size": path.stat().st_size},
        quality={
            "quality_score": score,
            "flags": flags,
            "auto_reject": auto_reject,
            "reject_reason": reason,
        },
    )


def build_group_session(tmp_path, raw_groups, enabled=True, strength="standard"):
    app = import_app_module()
    return app.build_session_from_groups(
        str(tmp_path),
        dry_run=True,
        mode="copy",
        raw_groups=raw_groups,
        infos=[info for group in raw_groups for info in group],
        threshold_near=10,
        threshold_far=6,
        near_seconds=300,
        prescreen_enabled=enabled,
        prescreen_strength=strength,
    )


def build_prescreen_session(app, tmp_path, infos, enabled=True, strength="standard"):
    app.LAST_INFOS = infos
    return app.build_prescreen_session_from_infos(
        str(tmp_path),
        dry_run=True,
        mode="copy",
        infos=infos,
        threshold_near=10,
        threshold_far=6,
        near_seconds=300,
        prescreen_enabled=enabled,
        prescreen_strength=strength,
    )


def wait_for_grouping(app, timeout=5.0):
    deadline = time.time() + timeout
    while app._GROUPING["status"] == "running" and time.time() < deadline:
        time.sleep(0.05)
    assert app._GROUPING["status"] == "done", app._GROUPING


def test_single_absolute_fatal_image_goes_to_losers_not_winners(tmp_path):
    bad = make_info(
        tmp_path / "bad.jpg",
        score=8,
        auto_reject=True,
        reason="画面缺少内容",
        flags=["tiny_file"],
    )

    sess = build_group_session(tmp_path, [[bad]])
    group = sess.groups[0]

    assert group.finished is True
    assert group.winner is None
    assert group.losers == [bad.path]
    assert group.auto_rejected == [bad.path]
    assert group.auto_reject_reasons[bad.path] == "画面缺少内容"


def test_prescreen_session_collects_rejected_photos_before_grouping(tmp_path):
    app = import_app_module()
    bad = make_info(tmp_path / "bad.jpg", score=5, auto_reject=True, reason="曝光过低")
    good = make_info(tmp_path / "good.jpg", score=72)

    sess = build_prescreen_session(app, tmp_path, [bad, good])

    assert sess.groups == []
    assert sess.prescreen_rejected == [bad.path]
    assert sess.prescreen_reject_reasons[bad.path] == "曝光过低"
    assert sess.prescreen_reviewed is False
    assert set(sess.meta) == {bad.path, good.path}


def test_standard_group_prescreen_rejects_clear_lowest_but_keeps_tournament(tmp_path):
    best = make_info(tmp_path / "best.jpg", score=94)
    ok = make_info(tmp_path / "ok.jpg", score=62)
    weak = make_info(tmp_path / "weak.jpg", score=55)

    sess = build_group_session(tmp_path, [[best, ok, weak]], enabled=True, strength="standard")

    group = sess.groups[0]
    assert group.finished is False
    assert group.left == best.path
    assert group.right == ok.path
    assert group.losers == [weak.path]
    assert group.auto_rejected == [weak.path]
    assert group.auto_reject_reasons[weak.path] == "同组美学/质量评分明显落后"


def test_disabled_prescreen_keeps_original_multi_group_flow(tmp_path):
    bad = make_info(tmp_path / "bad.jpg", score=5, auto_reject=True, reason="严重模糊")
    good = make_info(tmp_path / "good.jpg", score=90)

    sess = build_group_session(tmp_path, [[bad, good]], enabled=False)

    group = sess.groups[0]
    assert group.finished is False
    assert group.left == bad.path
    assert group.right == good.path
    assert group.losers == []
    assert group.auto_rejected == []


def test_auto_rejected_api_lists_rejected_items(tmp_path):
    app = import_app_module()
    bad = make_info(tmp_path / "bad.jpg", score=8, auto_reject=True, reason="严重模糊")
    app.SESSION = build_prescreen_session(app, tmp_path, [bad])

    payload = app.api_auto_rejected()

    assert payload["items"][0]["path"] == bad.path
    assert payload["items"][0]["group_id"] == "__prescreen__"
    assert payload["items"][0]["reason"] == "严重模糊"
    assert payload["items"][0]["restored"] is False


def test_restore_prescreen_rejected_marks_restored_once(tmp_path):
    app = import_app_module()
    bad = make_info(tmp_path / "bad.jpg", score=8, auto_reject=True, reason="严重模糊")
    app.SESSION = build_prescreen_session(app, tmp_path, [bad])
    app.request.get_json = lambda *args, **kwargs: {
        "group_id": "__prescreen__",
        "path": bad.path,
    }

    first = app.api_restore_rejected()
    second = app.api_restore_rejected()

    assert first["ok"] is True
    assert second["ok"] is True
    assert app.SESSION.prescreen_restored == [bad.path]


def test_confirm_prescreen_marks_session_reviewed(tmp_path):
    app = import_app_module()
    bad = make_info(tmp_path / "bad.jpg", score=8, auto_reject=True, reason="严重模糊")
    app.SESSION = build_group_session(tmp_path, [[bad]])

    payload = app.api_confirm_prescreen()

    assert payload["ok"] is True
    assert app.SESSION.prescreen_reviewed is True


def test_confirm_prescreen_groups_only_passed_and_restored_photos(tmp_path):
    app = import_app_module()
    bad_drop = make_info(tmp_path / "drop.jpg", score=8, auto_reject=True, reason="严重模糊")
    bad_restore = make_info(tmp_path / "restore.jpg", score=9, auto_reject=True, reason="曝光过低")
    good = make_info(tmp_path / "good.jpg", score=88)
    infos = [bad_drop, bad_restore, good]
    app.SESSION = build_prescreen_session(app, tmp_path, infos)
    app.request.get_json = lambda *args, **kwargs: {"group_id": "__prescreen__", "path": bad_restore.path}

    restored = app.api_restore_rejected()
    confirmed = app.api_confirm_prescreen()
    wait_for_grouping(app)

    assert restored["ok"] is True
    assert confirmed["ok"] is True
    all_group_images = [p for group in app.SESSION.groups for p in group.images]
    assert good.path in all_group_images
    assert bad_restore.path in all_group_images
    assert bad_drop.path in all_group_images  # kept only as an auto-rejected loser record
    tournament_images = [
        p
        for group in app.SESSION.groups
        if not group.auto_rejected
        for p in group.images
    ]
    assert good.path in tournament_images
    assert bad_restore.path in tournament_images
    assert bad_drop.path not in tournament_images
    assert app.SESSION.prescreen_reviewed is True
