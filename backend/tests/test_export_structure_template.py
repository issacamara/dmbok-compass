import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.contracts import StructureTemplate
from app.corpus.staging import validate_staging_files


ROOT = Path(__file__).parents[2]


def reviewed_map():
    return {
        "schema_version": "reviewed-structure-map-v1",
        "review_status": "approved",
        "source": {"sha256": "4d14b28ac6adb71b9551e66e109666fa6b467d525fd6d57704cb88dd9c8d76b4"},
        "structure_map": {
            "map_id": "reviewed-map",
            "entries": [{"section_path": ["Introduction"], "page_start": 1, "page_end": 628}],
        },
    }


def run_export(source, output, title="DAMA-DMBOK"):
    return subprocess.run(
        [sys.executable, "scripts/export_structure_template.py", str(source), "--document-title", title, "--output", str(output)],
        cwd=ROOT, env={"PYTHONPATH": "backend"}, capture_output=True, text=True, check=False,
    )


def test_exported_reviewed_map_passes_staging_check(tmp_path):
    source = tmp_path / "reviewed.json"
    original = json.dumps(reviewed_map())
    source.write_text(original)
    output = tmp_path / "template.json"
    result = run_export(source, output)
    assert result.returncode == 0, result.stderr
    data = output.read_bytes()
    template = StructureTemplate.model_validate_json(data)
    assert template.document_title == "DAMA-DMBOK"
    assert template.entries[0].page_end == 628
    assert set(json.loads(data)) == {"schema_version", "document_title", "instructions", "entries"}
    assert source.read_text() == original
    validate_staging_files(pdf_filename="DAMA-DMBOK.pdf", pdf_content_type="application/pdf", pdf_bytes=b"%PDF-1.7\n", map_filename="template.json", map_content_type="application/json", map_bytes=data)
    assert run_export(source, source).returncode != 0
    assert source.read_text() == original


@pytest.mark.parametrize("change,title", [
    ({"structure_map": {}}, "DAMA-DMBOK"),
    ({"structure_map": {"entries": [{"section_path": ["Bad"], "page_start": 2, "page_end": 1}]}}, "DAMA-DMBOK"),
    ({"schema_version": "structure-map-proposal-v1"}, "DAMA-DMBOK"),
    ({}, "  "),
])
def test_invalid_input_does_not_produce_output(tmp_path, change, title):
    source = tmp_path / "reviewed.json"
    source.write_text(json.dumps(reviewed_map() | change))
    output = tmp_path / "template.json"
    assert run_export(source, output, title).returncode != 0
    assert not output.exists()
