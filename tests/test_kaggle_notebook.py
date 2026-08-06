"""Tests for ``scripts/build_kaggle_notebook.py`` and the generated notebook.

The notebook only orchestrates already-tested repository APIs, so these tests
check two things: the notebook's *structure* (ordered, stage-tagged cells;
no embedded secrets) by reading it statically, and its *behavior* by actually
executing the cells tagged ``cpu_smoke`` with fake model adapters and a
temporary output directory -- real code paths, no GPU and no network.
"""

from __future__ import annotations

import ast
import json
import sys
import types
import zipfile
from pathlib import Path
from typing import Any

import nbformat
import pytest

from secure_rag_bench.evaluation.model_adapters import GenerationRequest, GenerationResult
from secure_rag_bench.evaluation.study_reporting import validate_study_bundle

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "securerag_native_adaptive_kaggle.ipynb"

EXPECTED_STAGE_ORDER = ["preflight", "pilot", "full_native", "ast_adaptive", "export"]


# ---------------------------------------------------------------------------
# Step 1: structural test (verbatim from the implementation plan)
# ---------------------------------------------------------------------------


def test_notebook_has_ordered_guarded_stages_and_no_embedded_secrets() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    sources = "\n".join(cell.source for cell in notebook.cells)
    assert [cell.metadata.get("stage") for cell in notebook.cells if cell.metadata.get("stage")] == [
        "preflight",
        "pilot",
        "full_native",
        "ast_adaptive",
        "export",
    ]
    assert "evaluate_validity_gate" in sources
    assert "if not gate.passed" in sources
    assert "sk-" not in sources and "ANTHROPIC_API_KEY=" not in sources


# ---------------------------------------------------------------------------
# Additional structural checks for the plan's explicit safety requirements
# ---------------------------------------------------------------------------


def test_stage_cells_are_unique_and_exactly_five() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    stages = [cell.metadata.get("stage") for cell in notebook.cells if cell.metadata.get("stage")]
    assert len(stages) == len(EXPECTED_STAGE_ORDER)
    assert stages == EXPECTED_STAGE_ORDER


_VALUE_RETURNING_METHOD_NAMES = frozenset({"write_text", "write_bytes"})


def test_no_code_cell_ends_with_a_bare_value_returning_call() -> None:
    """``Path.write_text``/``write_bytes`` return the count of chars/bytes written.

    If a call like that is the last top-level statement in a cell (not
    assigned to anything), Jupyter's displayhook auto-displays the return
    value as a confusing, undocumented trailing ``Out[]`` value underneath
    the cell's real printed output (e.g. a stray ``481`` under the preflight
    summary). Statements nested inside a compound statement (``for``/``if``)
    are never auto-displayed regardless of their value, so only the cell's
    final top-level statement matters. Guard against this for every code
    cell, not just the one instance already found and fixed.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        module = ast.parse(cell.source)
        if not module.body:
            continue
        last_statement = module.body[-1]
        if not isinstance(last_statement, ast.Expr) or not isinstance(last_statement.value, ast.Call):
            continue
        called_method = getattr(last_statement.value.func, "attr", None)
        assert called_method not in _VALUE_RETURNING_METHOD_NAMES, (
            f"cell tagged {cell.metadata.get('stage') or cell.metadata.get('tags')} ends with a bare, "
            f"unassigned call to `.{called_method}(...)`, which Jupyter will auto-display as a stray value"
        )


def test_setup_cell_does_not_clone_directly_into_the_working_directory() -> None:
    """``/kaggle/working`` is not guaranteed empty, so ``git clone <url> .`` is unsafe.

    Kaggle may seed the working directory with kernel metadata or an
    autosaved notebook copy before this cell ever runs, which makes a
    direct-into-``.`` clone fail with "destination path '.' already exists
    and is not an empty directory." The setup cell must clone into a
    scratch path and copy the contents over instead.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(cell for cell in notebook.cells if cell.get("id") == "cell-setup")
    assert "git clone" in cell.source
    assert "git clone --depth 1 https://github.com/malkarichayan1/SecureRag-Bench.git ." not in cell.source
    assert "cp -a" in cell.source


def test_setup_cell_clones_the_branch_that_actually_has_the_study_code() -> None:
    """Cloning without ``--branch`` checks out the repository's *default* branch.

    This repository's default branch (``main``) holds only planning docs --
    the study implementation (``native_cases.py`` and everything else the
    later cells import) lives on a feature branch. A clone missing
    ``--branch`` silently succeeds but produces a checkout that fails several
    cells later with a confusing ``ModuleNotFoundError``, not an obvious
    clone error.
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(cell for cell in notebook.cells if cell.get("id") == "cell-setup")
    assert "--branch" in cell.source
    assert 'SOURCE_GIT_REF = "codex/native-validity-adaptive"' in cell.source


def test_setup_cell_fails_fast_if_the_checkout_is_missing_study_code() -> None:
    """A silently-failed clone/install must not be discovered several cells later."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = next(cell for cell in notebook.cells if cell.get("id") == "cell-setup")
    assert "native_cases.py" in cell.source
    assert "raise RuntimeError(" in cell.source


def test_preflight_cell_prints_only_credential_presence_booleans() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = _cell_for_stage(notebook, "preflight")
    assert "credential presence (booleans only" in cell.source
    assert "bool(os.environ.get(" in cell.source
    # A credential's own value must never be interpolated into printed text.
    assert 'os.environ.get(_name)}"' not in cell.source
    assert "sk-" not in cell.source


def test_run_cells_use_jsonl_checkpoint_store() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for stage in ("pilot", "full_native"):
        cell = _cell_for_stage(notebook, stage)
        assert "JsonlCheckpointStore" in cell.source, f"{stage} cell must use JsonlCheckpointStore"


def test_full_native_cell_raises_before_selecting_a_generator() -> None:
    """The gate guard must appear textually before ``run_local_injecagent`` is called."""
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    cell = _cell_for_stage(notebook, "full_native")
    raise_index = cell.source.index('raise RuntimeError("validity gate not passed")')
    call_index = cell.source.index("run_local_injecagent(")
    assert raise_index < call_index


def test_no_notebook_cell_assigns_a_credential_value() -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        assert "ANTHROPIC_API_KEY=" not in cell.source
        assert "OPENAI_API_KEY=" not in cell.source
        assert "sk-" not in cell.source
        assert "hf_" not in cell.source


def _cell_for_stage(notebook: Any, stage: str) -> Any:
    for cell in notebook.cells:
        if cell.metadata.get("stage") == stage:
            return cell
    raise AssertionError(f"no cell tagged stage={stage!r}")


# ---------------------------------------------------------------------------
# Step 2 (builder correctness): rebuilding is deterministic
# ---------------------------------------------------------------------------


def test_build_notebook_is_deterministic(tmp_path: Path) -> None:
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_kaggle_notebook

    first = tmp_path / "a.ipynb"
    second = tmp_path / "b.ipynb"
    build_kaggle_notebook.build_notebook(first)
    build_kaggle_notebook.build_notebook(second)

    # Full-file byte equality, not just cell.source: a per-cell random id
    # (nbformat's default when none is passed) would pass a source-only
    # comparison while still making the written .ipynb non-reproducible.
    assert first.read_bytes() == second.read_bytes()
    first_notebook = nbformat.read(first, as_version=4)
    second_notebook = nbformat.read(second, as_version=4)
    assert nbformat.writes(first_notebook) == nbformat.writes(second_notebook)


def test_rebuilding_the_committed_notebook_produces_no_diff(tmp_path: Path) -> None:
    """The committed .ipynb must be exactly what the builder produces today."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_kaggle_notebook

    rebuilt = tmp_path / "rebuilt.ipynb"
    build_kaggle_notebook.build_notebook(rebuilt)

    assert rebuilt.read_bytes() == NOTEBOOK.read_bytes()


# ---------------------------------------------------------------------------
# Step 5: cpu_smoke execution harness (fake adapters, temporary paths)
# ---------------------------------------------------------------------------


class FixedTextGenerator:
    """A minimal ``TextGenerator`` fake: always returns the same completion."""

    def __init__(self, text: str, *, model_revision: str = "fake-rev") -> None:
        self.text = text
        self.model_revision = model_revision

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self.text


class FixedModelAdapter:
    """A minimal ``ModelAdapter`` fake for the AST plan-generation cell."""

    def __init__(self, text: str) -> None:
        self.text = text

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return GenerationResult(text=self.text, metadata={"provider": "fixed_fake"})


def _cpu_smoke_cells(notebook: Any) -> list[Any]:
    return [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "cpu_smoke" in cell.metadata.get("tags", [])
    ]


def _exec_cell(cell: Any, namespace: dict[str, Any]) -> None:
    exec(compile(cell.source, f"<cell:{cell.metadata.get('stage', 'setup')}>", "exec"), namespace)


def _base_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setenv("SECURE_RAG_BENCH_OUTPUT", str(tmp_path / "study"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.chdir(ROOT)
    return {
        "__name__": "__cpu_smoke__",
        "SELECTED_MODEL_NAMES": ["qwen-7b"],
        "TEST_GENERATOR_FACTORY": lambda catalog_entry: FixedTextGenerator(
            "Final Answer: I will safely answer the user request."
        ),
        "TEST_ADAPTER_FACTORY": lambda catalog_entry: FixedModelAdapter(
            "send_email('team@example.test', 'Status', 'On track')"
        ),
    }


def test_cpu_smoke_full_pipeline_produces_a_valid_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = _base_namespace(tmp_path, monkeypatch)
    cells = _cpu_smoke_cells(notebook)

    for cell in cells:
        if cell.metadata.get("stage") == "full_native":
            # Restrict the full stage to the held-out + calibration cases so
            # the smoke test stays fast without weakening bundle validity.
            namespace["FULL_STAGE_CASE_IDS"] = namespace["held_out_ids"] + namespace["calibration_case_ids"]
        _exec_cell(cell, namespace)

    assert namespace["configurations"]["qwen-7b"]["status"] == "completed"
    zip_path = namespace["STUDY_BUNDLE_ZIP"]
    assert Path(zip_path).exists()

    with zipfile.ZipFile(zip_path) as archive:
        extracted = tmp_path / "extracted"
        archive.extractall(extracted)
    bundle = validate_study_bundle(extracted)
    assert any(configuration.status == "completed" for configuration in bundle.configurations)


def test_cpu_smoke_full_native_raises_when_pilot_gate_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = _base_namespace(tmp_path, monkeypatch)
    # Text that never produces a valid protocol-parseable response, so every
    # held-out case comes back invalid and the gate fails.
    # Five words or fewer is the official InjecAgent parser's own "too short"
    # rejection, which the diagnostic maps to protocol_valid=False.
    namespace["TEST_GENERATOR_FACTORY"] = lambda catalog_entry: FixedTextGenerator("too short")

    cells = _cpu_smoke_cells(notebook)
    pilot_index = next(index for index, cell in enumerate(cells) if cell.metadata.get("stage") == "pilot")
    for cell in cells[: pilot_index + 1]:
        _exec_cell(cell, namespace)

    assert namespace["configurations"]["qwen-7b"]["gate"].passed is False

    full_native_cell = next(cell for cell in cells if cell.metadata.get("stage") == "full_native")
    with pytest.raises(RuntimeError, match="validity gate not passed"):
        _exec_cell(full_native_cell, namespace)


def test_cpu_smoke_preflight_cell_never_prints_a_fake_secret_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = _base_namespace(tmp_path, monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "totally-secret-value-should-not-print")

    preflight_cell = next(cell for cell in notebook.cells if cell.metadata.get("stage") == "preflight")
    _exec_cell(preflight_cell, namespace)

    captured = capsys.readouterr()
    assert "totally-secret-value-should-not-print" not in captured.out
    assert "ANTHROPIC_API_KEY: True" in captured.out


class _FakeUserSecretsClient:
    """Stands in for ``kaggle_secrets.UserSecretsClient`` in tests.

    Real behavior being modeled: attaching a secret in Kaggle's Secrets panel
    does not create an environment variable -- it only becomes retrievable
    through this client, and ``get_secret`` raises for a name that was never
    attached.
    """

    def __init__(self, secrets: dict[str, str]) -> None:
        self._secrets = secrets

    def get_secret(self, name: str) -> str:
        if name not in self._secrets:
            raise LookupError(f"No user secret exists for key '{name}'")
        return self._secrets[name]


def test_cpu_smoke_preflight_cell_bridges_attached_kaggle_secrets_into_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reproduces a real gap: attaching a Kaggle Secret alone leaves ``os.environ`` empty.

    Without a bridge, ``credential_presence`` reports ``False`` for a token
    the user genuinely attached in Kaggle's UI. The preflight cell must pull
    any attached secret into ``os.environ`` itself, without ever printing
    the value, and must tolerate secrets that were never attached (they stay
    absent, not an error).
    """
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = _base_namespace(tmp_path, monkeypatch)

    fake_module = types.ModuleType("kaggle_secrets")
    fake_module.UserSecretsClient = lambda: _FakeUserSecretsClient(  # type: ignore[attr-defined]
        {"HF_TOKEN": "fake-hf-secret-value-should-not-print"}
    )
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_module)

    preflight_cell = next(cell for cell in notebook.cells if cell.metadata.get("stage") == "preflight")
    try:
        _exec_cell(preflight_cell, namespace)

        import os

        assert os.environ["HF_TOKEN"] == "fake-hf-secret-value-should-not-print"
        assert namespace["credential_presence"] == {
            "ANTHROPIC_API_KEY": False,
            "OPENAI_API_KEY": False,
            "HF_TOKEN": True,
        }
        captured = capsys.readouterr()
        assert "fake-hf-secret-value-should-not-print" not in captured.out
        assert "HF_TOKEN: True" in captured.out
    finally:
        # The preflight cell mutates real os.environ directly (that's the
        # behavior under test), bypassing monkeypatch's own tracking -- undo
        # it explicitly so this fake secret can't leak into later tests.
        monkeypatch.delenv("HF_TOKEN", raising=False)


def test_cpu_smoke_pilot_writes_manifest_and_gate_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    namespace = _base_namespace(tmp_path, monkeypatch)
    cells = _cpu_smoke_cells(notebook)
    pilot_index = next(index for index, cell in enumerate(cells) if cell.metadata.get("stage") == "pilot")

    for cell in cells[: pilot_index + 1]:
        _exec_cell(cell, namespace)

    info = namespace["configurations"]["qwen-7b"]
    assert info["status"] == "completed_pilot"
    manifest_path = namespace["OUTPUT_ROOT"] / info["pilot"]["manifest"]
    assert manifest_path.exists()
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["payload"]["expected_case_ids"] == namespace["held_out_ids"]
