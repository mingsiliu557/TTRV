"""Convert a local, checksum-verified official TSV without loading its images in RAM.

No downloads, model imports, image re-encoding, annotation changes or filtering.
Run as python -m scripts.official_data.prepare from the verl directory.
"""
import argparse
import base64
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import random
import sqlite3
import string

REVISION = "4d0dc8096b2ee365491c469088cc462ae1fb6041"
SOURCES = {
    "AI2D_TEST": {"md5": "0f593e0d1c7df9a3d69bf1f947e71975",
                  "url": "https://opencompass.openxlab.space/utils/VLMEval/AI2D_TEST.tsv"},
    "MME": {"md5": "b36b43c3f09801f5d368627fb92187c3",
            "url": "https://opencompass.openxlab.space/utils/VLMEval/MME.tsv"},
}


def digest(path, algorithm="sha256"):
    h = hashlib.new(algorithm)
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def read_tsv(path):
    csv.field_size_limit(512 * 1024 * 1024)
    with open(path, encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"index", "question", "answer", "image"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing official fields: {required - set(reader.fieldnames or [])}")
        yield from reader


def prepare(tsv, output, dataset, subset_seed=42, subset_size=20):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from scripts.official_data.prompt_bridge import build_prompt

    tsv, output = Path(tsv).resolve(), Path(output).resolve()
    if not tsv.is_file():
        raise FileNotFoundError(f"Official local TSV required: {tsv}; legacy converted data is not a substitute")
    if digest(tsv, "md5") != SOURCES[dataset]["md5"]:
        raise ValueError(f"Official {dataset} MD5 mismatch; refusing version/mask substitution")
    output.mkdir(parents=True, exist_ok=False)
    (output / "images").mkdir()
    db = sqlite3.connect(output / "source_index.sqlite")
    db.execute("CREATE TABLE records (id TEXT PRIMARY KEY, row_json TEXT, image_path TEXT, image_hash TEXT, ref TEXT, ordinal INTEGER)")
    # First pass stores small metadata + image locations on disk. References may point forward.
    for ordinal, row in enumerate(read_tsv(tsv)):
        ident = row["index"]
        if not ident or not row["question"] or not row["answer"]:
            raise ValueError(f"Empty required field: {ident!r}")
        image = row.pop("image")
        metadata = {k: v for k, v in row.items() if v != ""}
        if len(image) < 64:  # official ImageBaseDataset short-value image-ID references
            path, image_hash, ref = None, None, image
        else:
            payload = base64.b64decode(image, validate=True)
            image_hash = hashlib.sha256(payload).hexdigest()
            path = f"images/{image_hash}.img"
            target = output / path
            if not target.exists():
                target.write_bytes(payload)
            del payload
            ref = None
        try:
            db.execute("INSERT INTO records VALUES (?,?,?,?,?,?)", (ident, json.dumps(metadata), path, image_hash, ref, ordinal))
        except sqlite3.IntegrityError as error:
            raise ValueError(f"Duplicate official ID {ident}") from error
    db.commit()
    ids = [r[0] for r in db.execute("SELECT id FROM records ORDER BY id")]
    if len(ids) < subset_size:
        raise ValueError("Official dataset smaller than adaptation subset")
    train_ids = random.Random(subset_seed).sample(ids, subset_size)
    # Sampling is stable over sorted IDs; dataset rows/pair positions retain source order.
    source_ids = [r[0] for r in db.execute("SELECT id FROM records ORDER BY ordinal")]
    protocol = {"dataset": dataset, "revision": REVISION, **SOURCES[dataset],
                "source_sha256": digest(tsv), "use_cot": False, "use_mpo": False,
                "scorer": "VLMEvalKit pure scoring definitions, exact_matching", "external_judge": False,
                "adaptation_unit": "question", "adaptation_source": "evaluation_pool",
                "subset_seed": subset_seed, "subset_size": subset_size,
                "prompt_bridge_sha256": digest(Path(__file__).with_name("prompt_bridge.py")),
                "upstream_scoring_sha256": digest(Path(__file__).parents[2] / "verl/third_party/official_vlmeval/scoring.py")}
    protocol_id = canonical_hash(protocol)
    pair_counts = Counter()
    writer = None
    train_rows = {}
    smoke_rows = []
    try:
        with open(output / "official_records.jsonl", "x", encoding="utf-8") as official:
            for ident in source_ids:
                row_json, image_path, image_hash, ref = db.execute(
                    "SELECT row_json,image_path,image_hash,ref FROM records WHERE id=?", (ident,)).fetchone()
                seen = {ident}
                while ref is not None:
                    if ref in seen:
                        raise ValueError(f"Cyclic image reference: {ident}")
                    seen.add(ref)
                    resolved = db.execute("SELECT image_path,image_hash,ref FROM records WHERE id=?", (ref,)).fetchone()
                    if resolved is None:
                        raise ValueError(f"Missing image reference {ref} from {ident}")
                    image_path, image_hash, ref = resolved
                raw = json.loads(row_json)
                choices = [c for c in string.ascii_uppercase if c in raw]
                if dataset == "AI2D_TEST" and (not choices or not set(choices) <= set("ABCD") or raw["answer"] not in choices):
                    raise ValueError(f"Unsupported AI2D label domain at {ident}: {choices}")
                if dataset == "MME" and raw["answer"].lower() not in {"yes", "no"}:
                    raise ValueError(f"Invalid official MME answer at {ident}")
                if dataset == "MME" and not {"category", "image_path"} <= raw.keys():
                    raise ValueError("MME requires original category and image_path for pair scoring")
                pair = f"{raw.get('category', '')}/{raw.get('image_path', image_hash)}"
                position = pair_counts[pair]
                pair_counts[pair] += 1
                prompt = build_prompt(raw, dataset)
                info = dict(index=f"{dataset}:{ident}", dataset=dataset, official_index=ident,
                            protocol_id=protocol_id, source_revision=REVISION, source_record_hash=canonical_hash(raw),
                            split="test", subset_seed=subset_seed, image_id=image_hash,
                            image_relative_path=image_path, image_sha256=image_hash,
                            question_raw=raw["question"], prompt_hash=canonical_hash(prompt),
                            official_answer_raw=raw["answer"], answer_type="yesno" if dataset == "MME" else "choice",
                            choice_labels=choices, category=raw.get("category", ""), pair_id=pair, pair_position=position)
                row = dict(data_source="MME-YN-TTT" if dataset == "MME" else "GPQA-TTT",
                           prompt=[dict(role="user", content=prompt)], images=[dict(image=str(output / image_path))],
                           ability="visual_question_answering", reward_model=dict(style="rule", ground_truth=raw["answer"].lower() if dataset == "MME" else raw["answer"]),
                           extra_info=info)
                table = pa.Table.from_pylist([row])
                if writer is None:
                    # The row schema is preserved; no benchmark images enter Arrow.
                    writer = pq.ParquetWriter(output / "test.parquet", table.schema)
                writer.write_table(table)
                if ident in train_ids:
                    train_rows[ident] = row
                if dataset != "MME" and len(smoke_rows) < 4:
                    smoke_rows.append(row)
                elif dataset == "MME" and (pair in {x['extra_info']['pair_id'] for x in smoke_rows} or len({x['extra_info']['pair_id'] for x in smoke_rows}) < 2):
                    smoke_rows.append(row)
                official.write(json.dumps(dict(raw=raw, extra_info=info), ensure_ascii=False) + "\n")
    finally:
        if writer is not None:
            writer.close()
        db.close()
    if dataset == "MME" and any(n != 2 for n in pair_counts.values()):
        raise ValueError("Official MME has incomplete/non-binary question pairs")
    pq.write_table(pa.Table.from_pylist([train_rows[i] for i in train_ids]), output / "train.parquet")
    pq.write_table(pa.Table.from_pylist(smoke_rows), output / "smoke.parquet")
    for name, value in {"protocol.lock.json": dict(protocol, protocol_id=protocol_id),
                        "adaptation_ids.json": [f"{dataset}:{i}" for i in train_ids],
                        "manifest.json": dict(dataset=dataset, count=len(ids), protocol_id=protocol_id,
                                              ids=[f"{dataset}:{i}" for i in ids], pair_counts=dict(pair_counts))}.items():
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    return protocol_id


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--tsv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", choices=SOURCES, required=True)
    parser.add_argument("--subset-seed", type=int, default=42)
    args = parser.parse_args()
    print(prepare(args.tsv, args.output, args.dataset, args.subset_seed))


if __name__ == "__main__":
    main()
