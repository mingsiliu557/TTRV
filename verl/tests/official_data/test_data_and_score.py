"""Small data/scorer fixtures only. No torch, model, GPU or network imports."""
import base64
import csv
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.official_data import prepare as preparation
from scripts.official_data.prompt_bridge import build_prompt
from scripts.official_data.upstream import module
from scripts.official_eval.score_predictions import join_predictions, score


@pytest.mark.parametrize("text,expected", [("Yes, it is.", "Yes"), ("I think yes", "Yes"),
                                         ("yes and no", "Unknown"), ("", "Unknown"), ("no!", "No")])
def test_official_yesno(text, expected):
    assert module("yesno").YOrN_Extraction(text) == expected


def test_prompt_independent_of_environment(monkeypatch):
    monkeypatch.setenv("USE_COT", "1")
    row = dict(question="Which label?", A="A/B on the diagram", B="another label")
    assert build_prompt(row, "AI2D_TEST") == "<image>\n" + module("scoring").build_multi_choice_prompt(row, "AI2D_TEST")
    assert build_prompt({"question": "Is it red?"}, "MME") == "<image>\nIs it red? Answer the question using a single word or phrase."


def test_mme_full_category_fixture(tmp_path):
    categories = ["OCR", "artwork", "celebrity", "color", "count", "existence", "landmark", "position", "posters", "scene",
                  "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"]
    rows = [dict(category=c, image_path=str(i // 2), score=s) for c in categories for i, s in enumerate([1, 1, 1, 0])]
    path = tmp_path / "mme.jsonl"
    pd.DataFrame(rows).to_json(path, lines=True, orient="records")
    result = module("scoring").MME_rating(path).iloc[0]
    assert result["perception"] == 1250
    assert result["reasoning"] == 500
    assert result["OCR"] == 125


def test_coverage_fail_closed():
    official = [{"extra_info": {"index": "AI2D_TEST:1"}}]
    pred = dict(index="AI2D_TEST:1", response_raw="", extra_info={"protocol_id": "p"})
    assert len(join_predictions(official, [pred], "p")) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        join_predictions(official, [pred, pred], "p")
    with pytest.raises(ValueError, match="coverage"):
        join_predictions(official, [], "p")
    with pytest.raises(ValueError, match="protocol"):
        join_predictions(official, [pred], "other")


def fixture_tsv(tmp_path, monkeypatch, duplicate=False):
    # Tiny valid PNG stored as original encoded bytes, not a benchmark image.
    payload = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aWQAAAABJRU5ErkJggg==")
    tsv = tmp_path / "source.tsv"
    fields = ["index", "question", "answer", "A", "B", "image"]
    with tsv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for i in range(22):
            writer.writerow(dict(index=str(0 if duplicate else i), question="Which label?", answer="A", A="first", B="second",
                                 image=base64.b64encode(payload).decode() if i == 21 else "21"))
    monkeypatch.setitem(preparation.SOURCES, "AI2D_TEST", dict(md5=preparation.digest(tsv, "md5"), url="fixture"))
    return tsv, payload


def test_conversion_and_official_scoring(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    tsv, payload = fixture_tsv(tmp_path, monkeypatch)
    data = tmp_path / "data"
    protocol_id = preparation.prepare(tsv, data, "AI2D_TEST")
    rows = pq.read_table(data / "test.parquet").to_pylist()
    assert len(rows) == 22
    assert len(pq.read_table(data / "train.parquet")) == 20
    assert Path(rows[0]["images"][0]["image"]).read_bytes() == payload
    assert all(r["prompt"][0]["content"].count("<image>") == 1 for r in rows)
    other = tmp_path / "same_seed"
    preparation.prepare(tsv, other, "AI2D_TEST")
    assert (data / "adaptation_ids.json").read_bytes() == (other / "adaptation_ids.json").read_bytes()
    predictions = tmp_path / "pred.jsonl"
    with predictions.open("w") as stream:
        for i, row in enumerate(rows):
            stream.write(json.dumps(dict(index=row["extra_info"]["index"], extra_info={"protocol_id": protocol_id},
                                         response_raw="A" if i else "", prediction="A" if i else "unknown")) + "\n")
    result = score(data, predictions, tmp_path / "score", "base", "fixture")
    assert result[0]["Overall"] == pytest.approx(21 / 22)
    report = json.loads((tmp_path / "score/coverage_and_invalid.json").read_text())
    assert report["official_invalid_rate"] == pytest.approx(1 / 22)
    with pytest.raises(FileExistsError):
        score(data, predictions, tmp_path / "score", "base", "fixture")


def test_bad_source_rejected(tmp_path, monkeypatch):
    tsv, _ = fixture_tsv(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        preparation.prepare(tsv, tmp_path / "duplicates", "AI2D_TEST")
    monkeypatch.setitem(preparation.SOURCES, "AI2D_TEST", dict(md5="wrong", url="fixture"))
    with pytest.raises(ValueError, match="MD5"):
        preparation.prepare(tsv, tmp_path / "bad_hash", "AI2D_TEST")


def test_data_tools_do_not_import_torch():
    import subprocess
    result = subprocess.run([sys.executable, "-c", "import sys; from scripts.official_data import prompt_bridge; from scripts.official_eval import score_predictions; assert 'torch' not in sys.modules; assert 'ray' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_memory_guard_covers_job_not_only_task(tmp_path):
    from scripts.official_data.run_suite import memory_sample, LIMIT, RESERVE
    job = tmp_path / "job_123"
    task = job / "step_0/task_special"
    task.mkdir(parents=True)
    (task / "memory.current").write_text("1")
    (job / "memory.current").write_text(str(LIMIT - RESERVE + 1))
    _, violations = memory_sample([(task / "memory.current", None), (job / "memory.current", None)])
    assert violations and violations[0][0] == str(job / "memory.current")


def test_mme_conversion_preserves_native_answers_and_pairs(tmp_path, monkeypatch):
    import pyarrow.parquet as pq
    categories = ["OCR", "artwork", "celebrity", "color", "count", "existence", "landmark", "position", "posters", "scene",
                  "code_reasoning", "commonsense_reasoning", "numerical_calculation", "text_translation"]
    _, payload = fixture_tsv(tmp_path, monkeypatch)
    tsv = tmp_path / "mme.tsv"
    fields = ["index", "question", "answer", "category", "image_path", "image"]
    with tsv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields, delimiter="\t")
        writer.writeheader()
        for category_i, category in enumerate(categories):
            for i in range(4):
                writer.writerow(dict(index=str(category_i * 4 + i), question="Is it present?", answer="Yes" if i % 2 == 0 else "No",
                                     category=category, image_path=f"image_{i // 2}.png", image=base64.b64encode(payload).decode()))
    monkeypatch.setitem(preparation.SOURCES, "MME", dict(md5=preparation.digest(tsv, "md5"), url="fixture"))
    data = tmp_path / "mme_data"
    protocol = preparation.prepare(tsv, data, "MME")
    rows = pq.read_table(data / "test.parquet").to_pylist()
    assert len(rows) == 56
    assert all(r["data_source"] == "MME-YN-TTT" and r["extra_info"]["choice_labels"] == [] for r in rows)
    assert [r["extra_info"]["pair_position"] for r in rows[:4]] == [0, 1, 0, 1]
    predictions = tmp_path / "mme.pred.jsonl"
    with predictions.open("w") as stream:
        for i, row in enumerate(rows):
            response = row["extra_info"]["official_answer_raw"] if i % 4 != 3 else "unknown"
            stream.write(json.dumps(dict(index=row["extra_info"]["index"], response_raw=response,
                                         extra_info={"protocol_id": protocol})) + "\n")
    result = score(data, predictions, tmp_path / "mme_score", "base", "fixture")
    assert result[0]["total"] == 1750
