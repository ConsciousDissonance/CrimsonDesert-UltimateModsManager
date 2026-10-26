"""Regression tests for multi-archive mods combined with DDS overlays.

All fixtures are synthetic. No game installation, extracted resources, account,
or mod archive is needed. Run inside a CDUMM checkout with its test dependencies.
"""

from dataclasses import astuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from cdumm.archive.overlay_builder import OverlayEntry
from cdumm.archive.pathc_handler import (
    PathcFile,
    PathcHeader,
    get_path_hash,
    read_pathc,
    serialize_pathc,
    update_entry,
)
from cdumm.archive.transactional_io import TransactionalIO
from cdumm.engine import import_handler
from cdumm.engine.apply_engine import ApplyWorker


def _texture_index(*paths):
    """A small valid index with one DDS template and distinct entry metadata."""
    index = PathcFile(
        PathcHeader(0, 0, 128, 1, 0, 0, 0),
        [b"DDS " + bytes(124)],
        [],
        [],
        [],
    )
    for number, path in enumerate(paths, start=1):
        update_entry(index, path, 0, (number * 100, 2, 3, 4))
    return serialize_pathc(index)


def _entries(path):
    """Compare meanings rather than the ordering of serialized table rows."""
    index = read_pathc(path)
    return dict(zip(index.key_hashes, map(astuple, index.map_entries)))


def _apply_overlay(game, vanilla, transaction):
    """Exercise the real Apply routine without starting a Qt worker or game."""
    dds = b"DDS " + bytes(124)
    worker = SimpleNamespace(
        _game_dir=game,
        _vanilla_dir=vanilla,
        _overlay_entries=[(dds, {"entry_path": "ui/overlay.dds"})],
    )
    packed = OverlayEntry("ui/texture", "overlay.dds", 0, 128, 128, 1, (400, 2, 3, 4))
    ApplyWorker._update_pathc_for_overlay_inner(worker, transaction, [packed])


def _setup_indexes(tmp_path):
    game, vanilla = tmp_path / "game", tmp_path / "vanilla"
    base = _texture_index("/ui/texture/base.dds")
    for root in [game, vanilla]:
        (root / "meta").mkdir(parents=True)
        (root / "meta/0.pathc").write_bytes(base)
    return game, vanilla


@pytest.mark.parametrize("wrapper", ["CityContent", "Seattle Content 雪"])
def test_archive_group_priority_survives_filesystem_enumeration(tmp_path, monkeypatch, wrapper):
    """An extractor's directory enumeration must not scramble group precedence."""
    package, game = tmp_path / "package", tmp_path / "game"
    game.mkdir()
    groups = [package / wrapper / name for name in ["0200", "0201", "0202"]]
    for group in groups:
        group.mkdir(parents=True)
        (group / "0.paz").write_bytes(b"synthetic payload")
        (group / "0.pamt").write_bytes(b"synthetic index")

    # Filesystems may enumerate in creation order, hash order, or another order.
    # Force an unfavorable order so the test is portable across host filesystems.
    real_rglob = Path.rglob

    def enumerate_package(path, pattern):
        if path == package:
            return iter(reversed(groups))
        return real_rglob(path, pattern)

    monkeypatch.setattr(Path, "rglob", enumerate_package)
    import_handler.clear_assigned_dirs()
    try:
        remap = import_handler._detect_standalone_mod(package, game, snapshot=None)
        assigned = [
            int(remap[group.relative_to(package).as_posix()]) for group in groups
        ]
        assert assigned == sorted(assigned)
        assert len(set(assigned)) == len(groups)
    finally:
        import_handler.clear_assigned_dirs()


@pytest.mark.parametrize(
    "game_parts",
    [
        ("Steam Library", "steamapps", "common", "Crimson Desert"),
        ("Crimson Desert.app", "Contents", "Resources", "packages"),
        ("ゲーム 雪", "Crimson Desert"),
    ],
    ids=["steam-library", "macos-app-bundle", "unicode-path"],
)
def test_pending_index_uses_native_paths_under_supported_install_layouts(tmp_path, game_parts):
    """Use real host filesystem operations, including commit's file moves.

    Run on Windows, macOS, and Linux: the same relative archive key must find
    the pending file under each host's native absolute installation path.
    """
    game = tmp_path.joinpath(*game_parts)
    staging = game / "CDMods" / "staging 雪"
    live_index = game / "meta" / "0.pathc"
    live_index.parent.mkdir(parents=True)
    live_index.write_bytes(b"final index")
    transaction = TransactionalIO(game, staging)

    # Merely existing on disk does not make a file part of this transaction.
    pending_index = staging / "meta" / "0.pathc"
    pending_index.parent.mkdir(parents=True)
    pending_index.write_bytes(b"stale index")
    assert transaction.staged_path("meta/0.pathc") is None

    transaction.stage_file("meta/0.pathc", b"earlier pending index")
    assert transaction.staged_path("meta/0.pathc") == pending_index
    assert transaction.staged_files() == ["meta/0.pathc"]
    assert transaction.stage_file_if_changed("meta/0.pathc", b"final index")
    assert not transaction.stage_file_if_changed("meta/0.pathc", b"final index")
    transaction.commit()
    assert live_index.read_bytes() == b"final index"
    assert not pending_index.exists()
    assert not live_index.with_suffix(".pathc.pre-apply").exists()


def test_full_index_mod_and_overlay_keep_both_texture_sets(tmp_path):
    game, vanilla = _setup_indexes(tmp_path)
    full_index = _texture_index("/ui/texture/base.dds", "/object/city/wall.dds")
    transaction = TransactionalIO(game, tmp_path / "staging")
    transaction.stage_file("meta/0.pathc", full_index)
    staged_index = tmp_path / "staging/meta/0.pathc"
    expected = _entries(staged_index)

    _apply_overlay(game, vanilla, transaction)
    transaction.commit()
    actual = _entries(game / "meta/0.pathc")
    assert all(actual.get(key) == value for key, value in expected.items())
    assert len(actual) == len(expected) + 1


def test_repeated_apply_keeps_the_overlay_when_live_file_already_matches(tmp_path):
    game, vanilla = _setup_indexes(tmp_path)
    full_index = _texture_index("/ui/texture/base.dds", "/object/city/wall.dds")
    for number in range(2):
        transaction = TransactionalIO(game, tmp_path / f"staging-{number}")
        transaction.stage_file("meta/0.pathc", full_index)
        _apply_overlay(game, vanilla, transaction)
        transaction.commit()
        current = (game / "meta/0.pathc").read_bytes()
        if number == 0:
            first_result = current
        else:
            assert current == first_result


def test_deactivating_full_index_mod_removes_only_its_textures(tmp_path):
    game, vanilla = _setup_indexes(tmp_path)
    transaction = TransactionalIO(game, tmp_path / "enabled")
    transaction.stage_file(
        "meta/0.pathc",
        _texture_index("/ui/texture/base.dds", "/object/city/wall.dds"),
    )
    _apply_overlay(game, vanilla, transaction)
    transaction.commit()

    # With the city disabled there is no staged full index. Starting from the
    # live result would retain disabled registrations; start from vanilla instead.
    transaction = TransactionalIO(game, tmp_path / "disabled")
    _apply_overlay(game, vanilla, transaction)
    transaction.commit()
    actual = _entries(game / "meta/0.pathc")
    base = _entries(vanilla / "meta/0.pathc")
    overlay_hash = get_path_hash("/ui/texture/overlay.dds")
    assert set(actual) == set(base) | {overlay_hash}
    assert all(actual[key] == value for key, value in base.items())
    assert actual[overlay_hash][1:] == (400, 2, 3, 4)


def test_unregistered_index_left_in_staging_is_not_composed(tmp_path):
    game, vanilla = _setup_indexes(tmp_path)
    staging = tmp_path / "staging"
    stale_index = staging / "meta/0.pathc"
    stale_index.parent.mkdir(parents=True)
    stale_index.write_bytes(
        _texture_index("/ui/texture/base.dds", "/object/disabled/wall.dds")
    )
    # Files left by an earlier operation are not owned by this transaction.
    transaction = TransactionalIO(game, staging)
    _apply_overlay(game, vanilla, transaction)
    transaction.commit()
    actual = _entries(game / "meta/0.pathc")
    assert set(actual) == set(_entries(vanilla / "meta/0.pathc")) | {
        get_path_hash("/ui/texture/overlay.dds")
    }


@pytest.mark.parametrize("live_matches", [True, False])
def test_write_optimization_compares_pending_transaction_before_live_file(
    tmp_path, live_matches
):
    game = tmp_path / "game"
    game.mkdir()
    target = b"desired final result"
    (game / "data.bin").write_bytes(target if live_matches else b"old live data")
    transaction = TransactionalIO(game, tmp_path / "staging")
    transaction.stage_file("data.bin", b"earlier pending write")
    assert transaction.stage_file_if_changed("data.bin", target)
    transaction.commit()
    assert (game / "data.bin").read_bytes() == target


def test_identical_pending_write_is_skipped(tmp_path):
    game = tmp_path / "game"
    game.mkdir()
    (game / "data.bin").write_bytes(b"old live data")
    transaction = TransactionalIO(game, tmp_path / "staging")
    transaction.stage_file("data.bin", b"desired final result")
    assert not transaction.stage_file_if_changed("data.bin", b"desired final result")
    transaction.commit()
    assert (game / "data.bin").read_bytes() == b"desired final result"
