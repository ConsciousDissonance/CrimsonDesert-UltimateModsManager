"""Apply/import regressions for an installation that already contains mods.

The fixtures contain synthetic archives and a persisted SQLite database. Tests
run the real Apply pipeline, including its later revert and catalog phases.
"""

import struct
from dataclasses import astuple
from pathlib import Path

import pytest

from cdumm.archive.hashlittle import compute_pamt_hash
from cdumm.archive.overlay_builder import build_overlay
from cdumm.archive.papgt_manager import read_papgt_entries
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
from cdumm.engine.apply_engine import ApplyWorker, invalidate_apply_fingerprint
from cdumm.engine.delta_engine import generate_delta, save_entry_delta
from cdumm.engine.snapshot_manager import SnapshotManager, hash_file


def _index(*paths):
    index = PathcFile(PathcHeader(0, 0, 128, 1, 0, 0, 0), [b"DDS " + bytes(124)], [], [], [])
    for number, path in enumerate(paths, start=1):
        update_entry(index, path, 0, (100 * number, 2, 3, 4))
    return serialize_pathc(index)


def _index_entries(game):
    index = read_pathc(game / "meta" / "0.pathc")
    return dict(zip(index.key_hashes, map(astuple, index.map_entries)))


def _archive(label):
    """A real, parseable PAZ/PAMT pair containing one small raw HTML entry."""
    body = (f"<html>{label}</html>" * 100).encode()
    paz, pamt, _ = build_overlay(
        [(body, {"entry_path": f"ui/{label}.html", "compression_type": 0})]
    )
    return paz, pamt


def _catalog(groups):
    names = bytearray()
    rows = bytearray()
    for name, pamt in groups.items():
        rows += struct.pack("<III", 0x003FFF00, len(names), compute_pamt_hash(pamt))
        names += name.encode("ascii") + b"\0"
    return (
        b"\1\2\3\4" + bytes(4) + bytes([len(groups), 255, 255, 255])
        + rows + struct.pack("<I", len(names)) + names
    )


def _setup_game(tmp_path, db=None):
    game, vanilla = tmp_path / "game", tmp_path / "vanilla"
    groups = {}
    for name in ("0001", "0008"):
        paz, pamt = _archive(f"base_{name}")
        groups[name] = pamt
        for root in (game, vanilla):
            (root / name).mkdir(parents=True)
            (root / name / "0.paz").write_bytes(paz)
            (root / name / "0.pamt").write_bytes(pamt)
    for root in (game, vanilla):
        (root / "meta").mkdir()
        (root / "meta" / "0.pathc").write_bytes(_index("/ui/texture/base.dds"))
        (root / "meta" / "0.papgt").write_bytes(_catalog(groups))
    if db is not None:
        for path in game.rglob("*"):
            if path.is_file():
                digest, size = hash_file(path)
                db.connection.execute(
                    "INSERT INTO snapshots (file_path, file_hash, file_size) VALUES (?, ?, ?)",
                    (path.relative_to(game).as_posix(), digest, size),
                )
        db.connection.commit()
    return game, vanilla


def _record_mod(db, name, priority, files, *, enabled=True):
    """Store existing-format deltas without running a fresh import of that mod.

    Each file is (relative game path, saved delta bytes, is_new, entry_path).
    This models records that survive an application upgrade or restart.
    """
    mod_id = db.connection.execute(
        "INSERT INTO mods (name, mod_type, priority, enabled) VALUES (?, 'paz', ?, ?)",
        (name, priority, int(enabled)),
    ).lastrowid
    for number, (file_path, content, is_new, entry_path) in enumerate(files):
        delta = db.db_path.parent / "deltas" / str(mod_id) / f"{number}.delta"
        delta.parent.mkdir(parents=True, exist_ok=True)
        delta.write_bytes(content)
        db.connection.execute(
            "INSERT INTO mod_deltas "
            "(mod_id, file_path, delta_path, byte_start, byte_end, is_new, entry_path) "
            "VALUES (?, ?, ?, 0, ?, ?, ?)",
            (mod_id, file_path, str(delta), len(content), int(is_new), entry_path),
        )
    db.connection.commit()
    return mod_id


def _install_texture_mod(db, tmp_path, priority=10, entry="ui/texture/existing.dds"):
    header = bytearray(128)
    header[:4] = b"DDS "
    for offset, value in ((4, 124), (8, 0x21007), (12, 32), (16, 32), (20, 512),
                          (28, 1), (76, 32), (80, 4)):
        struct.pack_into("<I", header, offset, value)
    header[84:88] = b"DXT1"
    delta = tmp_path / "existing-texture.entr"
    save_entry_delta(
        bytes(header) + bytes([0xAB, 0xCD] * 256),
        {"entry_path": entry, "pamt_dir": "0008", "compression_type": 1},
        delta,
    )
    return _record_mod(db, "Existing texture mod", priority, [
        ("0008/0.paz", delta.read_bytes(), False, entry),
    ])


def _apply(game, vanilla, db):
    """Reopen the database in a fresh worker, and force the full pipeline.

    Removing the normal Apply shortcut ensures repeated-run assertions exercise
    composition/revert rather than merely reporting an unchanged fingerprint.
    """
    invalidate_apply_fingerprint(game)
    worker = ApplyWorker(game, vanilla, db.db_path)
    errors, finished = [], []
    worker.error_occurred.connect(errors.append)
    worker.finished.connect(lambda: finished.append(True))
    worker.run()
    assert errors == []
    assert finished == [True]
    return worker


@pytest.mark.parametrize("texture_path", ["ui/texture/existing.dds", "ui/texture/base.dds"])
def test_disabling_new_mod_keeps_preexisting_mods_through_complete_apply(tmp_path, db, texture_path):
    game, vanilla = _setup_game(tmp_path, db)
    _install_texture_mod(db, tmp_path, entry=texture_path)
    original = (game / "0001" / "0.paz").read_bytes()
    changed = bytearray(original)
    changed[20] ^= 1
    _record_mod(db, "Existing byte patch", 20, [
        ("0001/0.paz", generate_delta(original, bytes(changed)), False, None),
    ])
    old_paz, old_pamt = _archive("existing_archive")
    _record_mod(db, "Existing archive mod", 30, [
        ("0041/0.paz", old_paz, True, None),
        ("0041/0.pamt", old_pamt, True, None),
    ])
    # Another tool's archive is on disk but has no CDUMM database rows.
    external = game / "0048"
    external.mkdir()
    external_paz, external_pamt = _archive("external")
    (external / "0.paz").write_bytes(external_paz)
    (external / "0.pamt").write_bytes(external_pamt)

    _apply(game, vanilla, db)
    existing_index = _index_entries(game)
    assert get_path_hash("/" + texture_path) in existing_index

    city_paz, city_pamt = _archive("city")
    city = _record_mod(db, "New full-index archive mod", 5, [
        ("meta/0.pathc", b"FULL" + _index(
            "/ui/texture/base.dds", "/object/city/wall.dds"
        ), False, None),
        ("0050/0.paz", city_paz, True, None),
        ("0050/0.pamt", city_pamt, True, None),
    ])
    for enabled in (True, True, False, False, True, False, False):
        db.connection.execute("UPDATE mods SET enabled = ? WHERE id = ?", (int(enabled), city))
        db.connection.commit()
        _apply(game, vanilla, db)

        actual = _index_entries(game)
        expected_keys = set(existing_index)
        if enabled:
            expected_keys.add(get_path_hash("/object/city/wall.dds"))
        assert set(actual) == expected_keys
        assert all(actual[key] == value for key, value in existing_index.items())
        assert (game / "0001" / "0.paz").read_bytes() == bytes(changed)
        assert (game / "0041" / "0.paz").read_bytes() == old_paz
        assert (game / "0041" / "0.pamt").read_bytes() == old_pamt
        assert (external / "0.paz").read_bytes() == external_paz
        assert (external / "0.pamt").read_bytes() == external_pamt
        assert (game / "0050" / "0.pamt").exists() == enabled
        catalog = {name for name, _, _ in read_papgt_entries(game / "meta" / "0.papgt")}
        assert {"0001", "0008", "0041", "0048"} <= catalog
        assert ("0050" in catalog) == enabled


@pytest.mark.parametrize("existing_priority,new_priority", [(1, 5), (20, 5)])
def test_existing_full_index_conflicts_still_follow_load_order(
    tmp_path, db, existing_priority, new_priority
):
    """Full-index replacements still pick a winner; DDS overlays layer onto it."""
    game, vanilla = _setup_game(tmp_path, db)
    texture = _install_texture_mod(db, tmp_path)
    existing = _record_mod(db, "Existing index replacement", existing_priority, [
        ("meta/0.pathc", b"FULL" + _index(
            "/ui/texture/base.dds", "/object/existing/wall.dds"
        ), False, None),
    ])
    _apply(game, vanilla, db)
    new = _record_mod(db, "New index replacement", new_priority, [
        ("meta/0.pathc", b"FULL" + _index(
            "/ui/texture/base.dds", "/object/new/wall.dds"
        ), False, None),
    ])

    # Exercise both precedence orders, then disabling the lower-priority mod.
    # The losing full index must not be silently merged into the winning one.
    for old_priority in (existing_priority, 30, 1):
        db.connection.execute("UPDATE mods SET priority = ? WHERE id = ?", (old_priority, existing))
        db.connection.commit()
        _apply(game, vanilla, db)
        winning_path = "/object/existing/wall.dds" if old_priority < new_priority else (
            "/object/new/wall.dds"
        )
        assert set(_index_entries(game)) == {
            get_path_hash("/ui/texture/base.dds"),
            get_path_hash("/ui/texture/existing.dds"),
            get_path_hash(winning_path),
        }

    db.connection.execute("UPDATE mods SET enabled = 0 WHERE id = ?", (new,))
    db.connection.commit()
    _apply(game, vanilla, db)
    with_texture = _index_entries(game)
    assert get_path_hash("/object/existing/wall.dds") in with_texture
    assert get_path_hash("/object/new/wall.dds") not in with_texture

    # Turning off the DDS mod should remove its registration while keeping the
    # existing full-index mod intact; turning it back on restores that texture.
    db.connection.execute("UPDATE mods SET enabled = 0 WHERE id = ?", (texture,))
    db.connection.commit()
    _apply(game, vanilla, db)
    assert set(_index_entries(game)) == {
        get_path_hash("/ui/texture/base.dds"), get_path_hash("/object/existing/wall.dds")
    }
    db.connection.execute("UPDATE mods SET enabled = 1 WHERE id = ?", (texture,))
    db.connection.commit()
    _apply(game, vanilla, db)
    assert _index_entries(game) == with_texture


def test_import_after_restart_preserves_existing_archive_assignments(tmp_path, db, monkeypatch):
    game, _ = _setup_game(tmp_path, db)
    old_paz, old_pamt = _archive("old")
    for name, enabled in (("0036", True), ("0037", True), ("0038", False)):
        _record_mod(db, f"Existing {name}", int(name), [
            (f"{name}/0.paz", old_paz, True, None),
            (f"{name}/0.pamt", old_pamt, True, None),
        ], enabled=enabled)
    for name in ("0036", "0040"):
        (game / name).mkdir()
        (game / name / "0.paz").write_bytes(old_paz)
        (game / name / "0.pamt").write_bytes(old_pamt)

    # 0037 is imported but unapplied; 0038 belongs to a disabled mod. Both must
    # retain their slots even in a fresh worker without in-memory reservations.
    db.connection.execute("UPDATE mods SET applied = 1 WHERE name = 'Existing 0036'")
    db.connection.commit()
    old_rows = db.connection.execute("SELECT * FROM mod_deltas ORDER BY id").fetchall()
    old_mods = db.connection.execute("SELECT * FROM mods ORDER BY id").fetchall()
    old_payloads = {Path(row[3]): Path(row[3]).read_bytes() for row in old_rows}
    db.close()
    db.initialize()
    package = tmp_path / "new package"
    source_paz = {}
    for name in ("0202", "0201", "0200"):
        new_paz, new_pamt = _archive(f"new_{name}")
        source_paz[name] = new_paz
        group = package / "CityContent" / name
        group.mkdir(parents=True)
        (group / "0.paz").write_bytes(new_paz)
        (group / "0.pamt").write_bytes(new_pamt)

    real_rglob = Path.rglob

    def reverse_package(path, pattern):
        entries = list(real_rglob(path, pattern))
        return iter(reversed(entries)) if path == package else iter(entries)

    monkeypatch.setattr(Path, "rglob", reverse_package)
    import_handler.clear_assigned_dirs()
    try:
        result = import_handler.import_from_folder(
            package, game, db, SnapshotManager(db), tmp_path / "new-deltas"
        )
        assert result.error is None
        assert result.mod_id is not None
        new_files = db.connection.execute(
            "SELECT file_path, delta_path FROM mod_deltas WHERE mod_id = ?",
            (result.mod_id,),
        ).fetchall()
        new_dirs = {path.split("/")[0] for path, _ in new_files}
        assert len(new_dirs) == 3
        assert new_dirs.isdisjoint({"0036", "0037", "0038", "0040"})
        assert len(new_files) == 6
        payload_to_slot = {
            Path(delta).read_bytes(): int(path.split("/")[0])
            for path, delta in new_files if path.endswith(".paz")
        }
        slots = [payload_to_slot[source_paz[name]] for name in sorted(source_paz)]
        assert slots == sorted(slots)
        assert db.connection.execute(
            "SELECT * FROM mod_deltas WHERE mod_id != ? ORDER BY id", (result.mod_id,)
        ).fetchall() == old_rows
        assert db.connection.execute(
            "SELECT * FROM mods WHERE id != ? ORDER BY id", (result.mod_id,)
        ).fetchall() == old_mods
        assert all(path.read_bytes() == content for path, content in old_payloads.items())
        for name in ("0036", "0040"):
            assert (game / name / "0.paz").read_bytes() == old_paz
            assert (game / name / "0.pamt").read_bytes() == old_pamt
    finally:
        import_handler.clear_assigned_dirs()


def test_failed_commit_restores_preexisting_files_after_pending_index_replacement(
    tmp_path, monkeypatch
):
    import shutil

    game, _ = _setup_game(tmp_path)
    old_paz = (game / "0001" / "0.paz").read_bytes()
    old_index = (game / "meta" / "0.pathc").read_bytes()
    transaction = TransactionalIO(game, tmp_path / "staging")
    transaction.stage_file("0001/0.paz", b"changed archive")
    transaction.stage_file("meta/0.pathc", _index("/object/earlier/wall.dds"))
    assert transaction.stage_file_if_changed("meta/0.pathc", _index("/object/final/wall.dds"))

    real_move = shutil.move

    def fail_index_move(source, target):
        if Path(target) == game / "meta" / "0.pathc":
            raise OSError("synthetic file-move failure")
        return real_move(source, target)

    monkeypatch.setattr(shutil, "move", fail_index_move)
    with pytest.raises(OSError, match="synthetic file-move failure"):
        transaction.commit()
    assert (game / "0001" / "0.paz").read_bytes() == old_paz
    assert (game / "meta" / "0.pathc").read_bytes() == old_index
    assert list(game.rglob("*.pre-apply")) == []
