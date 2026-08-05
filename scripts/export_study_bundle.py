"""Export a validated study bundle directory into a portable, safe zip archive.

Usage::

    python scripts/export_study_bundle.py --run-root PATH --output PATH.zip

``--run-root`` must already be a directory matching the schema documented in
``secure_rag_bench.evaluation.study_reporting``'s module docstring (the
``bundle.json``/``MANIFEST.sha256``/``splits``/``checkpoints``/``runs``/...
tree). This script does **not** build a bundle from scratch or orchestrate a
study run -- producing ``--run-root`` is the Kaggle notebook's job. It only:

1. Validates ``--run-root`` through ``validate_study_bundle`` *before*
   touching the archive, so an invalid source fails fast instead of being
   exported as something ``scripts/import_study_bundle.py`` would later have
   to reject anyway.
2. Writes every regular file in the tree into the archive under a
   normalized, POSIX-relative arcname with a fixed (1980-01-01) timestamp,
   so the resulting zip is byte-for-byte reproducible across runs on the
   same input. Symlinks anywhere in the tree -- file or directory -- are
   rejected explicitly: ``Path.iterdir()``/``os.walk()`` will otherwise
   silently follow them into content outside the bundle root.

Nothing beyond the declared bundle schema is written to the archive: no
environment variables, no credentials, no machine-local paths from the
exporting host. Environment metadata (GPU/package/timestamp capture) is a
declared, redaction-checked part of the schema -- see point 12 of "What
validation actually checks" in ``study_reporting``'s module docstring -- and
is exported as-is because ``validate_study_bundle`` already rejects anything
credential-shaped inside it.
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from pathlib import Path

from secure_rag_bench.evaluation.study_reporting import (
    BundleValidationError,
    validate_study_bundle,
)

#: Zip entries get a fixed timestamp so two exports of the same bundle
#: produce byte-identical archives (real content differences still change
#: the archive; only the export's own wall-clock time is suppressed).
_FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

#: Unix "regular file, rw-r--r--" bits, stored in ``external_attr`` so tools
#: that inspect the archive on POSIX see ordinary file permissions rather
#: than the zeroed-out default zipfile otherwise writes.
_REGULAR_FILE_EXTERNAL_ATTR = 0o100644 << 16


class BundleExportError(ValueError):
    """Raised when ``--run-root`` cannot be safely turned into an archive."""


def _iter_bundle_files(root: Path) -> list[Path]:
    """Return every regular file under ``root``, rejecting any symlink.

    ``os.walk(followlinks=False)`` will not *descend into* a symlinked
    directory, but it still reports the symlink itself as a directory name
    in ``dirnames`` (and would silently skip it rather than reject it) and
    reports symlinked files in ``filenames`` unchanged. Both are checked
    explicitly here so a symlink anywhere in the tree is a hard failure
    instead of quietly-omitted or quietly-followed content.
    """
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        for dirname in dirnames:
            candidate = current / dirname
            if candidate.is_symlink():
                relative = candidate.relative_to(root).as_posix()
                raise BundleExportError(
                    f"{relative}: symlinked directory is not allowed in a bundle export"
                )
        for filename in filenames:
            candidate = current / filename
            if candidate.is_symlink():
                relative = candidate.relative_to(root).as_posix()
                raise BundleExportError(
                    f"{relative}: symlink is not allowed in a bundle export"
                )
            files.append(candidate)
    return files


def export_bundle(run_root: Path, output: Path) -> Path:
    """Validate ``run_root`` and write it to ``output`` as a safe zip archive.

    Raises ``BundleValidationError`` if ``run_root`` does not pass
    ``validate_study_bundle``, or ``BundleExportError`` if the tree contains
    a symlink. ``output`` is written atomically: a sibling ``.tmp`` file is
    built first and only renamed into place once the archive is complete, so
    a failure partway through never leaves a truncated file at ``output``.
    """
    run_root = Path(run_root)
    output = Path(output)

    # Fail fast: never zip a bundle that would not itself be importable.
    validate_study_bundle(run_root)

    files = sorted(
        _iter_bundle_files(run_root),
        key=lambda path: path.relative_to(run_root).as_posix(),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name + ".tmp")
    if staging.exists():
        staging.unlink()
    try:
        with zipfile.ZipFile(staging, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                arcname = path.relative_to(run_root).as_posix()
                info = zipfile.ZipInfo(arcname, date_time=_FIXED_ZIP_TIMESTAMP)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = _REGULAR_FILE_EXTERNAL_ATTR
                archive.writestr(info, path.read_bytes())
        os.replace(staging, output)
    finally:
        if staging.exists():
            staging.unlink()
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path, help="A validated study bundle directory")
    parser.add_argument("--output", required=True, type=Path, help="Destination .zip path")
    args = parser.parse_args(argv)

    try:
        export_bundle(args.run_root, args.output)
    except BundleValidationError as exc:
        print(f"error: --run-root failed bundle validation: {exc}", file=sys.stderr)
        return 1
    except BundleExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(str(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
