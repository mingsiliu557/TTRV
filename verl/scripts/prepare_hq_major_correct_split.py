#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
from copy import deepcopy
from pathlib import Path

from datasets import Dataset


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def write_dataset(path: Path, train_rows, test_rows) -> None:
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / 'train.json', train_rows)
    write_json(path / 'test.json', test_rows)
    Dataset.from_list(train_rows).to_parquet(str(path / 'train.parquet'))
    Dataset.from_list(test_rows).to_parquet(str(path / 'test.parquet'))


def iter_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def retag(row, dataset_name: str, split: str, new_index: int, source_index: str):
    item = deepcopy(row)
    extra = dict(item.get('extra_info') or {})
    extra['source_dataset'] = extra.get('dataset')
    extra['source_index'] = source_index
    extra['split'] = split
    extra['index'] = f'{dataset_name}-{split}-{new_index}'
    extra['dataset'] = dataset_name
    item['extra_info'] = extra
    return item


def normalize_choice_or_number(value):
    if value is None:
        return ''
    text = str(value).strip()
    if not text:
        return ''
    upper = text.upper()
    if re.fullmatch(r'[A-Z]', upper):
        return upper
    nums = re.findall(r'-?\d+(?:\.\d+)?', text.replace(',', ''))
    if nums:
        num = nums[-1]
        try:
            val = float(num)
            if val.is_integer():
                return str(int(val))
            return str(val)
        except Exception:
            return num
    return upper



def normalize_direct_answer(value):
    if value is None:
        return ''
    text = str(value).strip().lower()
    text = re.sub(r'<\|[^>]*\|>|</?s>|<pad>|<image>|<IMG_CONTEXT>', ' ', text, flags=re.IGNORECASE)
    text = text.replace('’', "'").replace('`', "'").replace(',', '')
    text = re.sub(r'[^0-9a-zA-Z]+', ' ', text)
    tokens = [tok for tok in text.split() if tok not in {'a', 'an', 'the'}]
    return ' '.join(tokens)


def is_match(pred, truth, metric: str, smape_threshold: float):
    pred_n = normalize_choice_or_number(pred)
    truth_n = normalize_choice_or_number(truth)
    if metric == 'exact':
        return pred_n == truth_n, {'normalized_prediction': pred_n, 'normalized_truth': truth_n}
    if metric == 'direct_answer':
        pred_d = normalize_direct_answer(pred)
        truth_d = normalize_direct_answer(truth)
        return pred_d == truth_d and bool(pred_d), {'normalized_prediction': pred_d, 'normalized_truth': truth_d}
    if metric == 'numeric_smape':
        try:
            p = float(pred_n)
            y = float(truth_n)
        except Exception:
            return False, {'normalized_prediction': pred_n, 'normalized_truth': truth_n, 'smape_score': 0.0}
        denom = abs(p) + abs(y)
        score = 1.0 if denom == 0 else 1.0 - abs(p - y) / denom
        return score >= smape_threshold, {'normalized_prediction': pred_n, 'normalized_truth': truth_n, 'smape_score': score}
    raise ValueError(f'unknown metric={metric}')


def sample_reward_value(sample):
    try:
        return float(sample.get('reward', 0.0))
    except Exception:
        return 0.0


def sample_is_correct(sample):
    return bool(sample.get('correct')) or sample_reward_value(sample) > 0.0


def best_correct_sample(record):
    correct_samples = []
    for sample in record.get('samples') or []:
        if sample_is_correct(sample):
            item = dict(sample)
            item['_reward_value'] = sample_reward_value(sample)
            correct_samples.append(item)
    if not correct_samples:
        return None
    return max(
        correct_samples,
        key=lambda item: (
            float(item.get('_reward_value', 0.0)),
            -int(item.get('sample_index', 10**9)),
        ),
    )


def make_pool(args):
    source_dir = Path(args.source_dir)
    source_test = load_json(source_dir / 'test.json')
    source_train = load_json(source_dir / 'train.json')
    if args.pool_size > len(source_test):
        raise ValueError(f'pool_size {args.pool_size} > source test size {len(source_test)}')
    rng = random.Random(args.seed)
    selected_positions = sorted(rng.sample(range(len(source_test)), args.pool_size))
    dataset_name = Path(args.output_dir).name
    train_rows = []
    selected = []
    for i, pos in enumerate(selected_positions):
        row = source_test[pos]
        src_index = (row.get('extra_info') or {}).get('index', str(pos))
        train_rows.append(retag(row, dataset_name, 'train', i, src_index))
        selected.append({'pool_index': i, 'source_position': pos, 'source_index': src_index})
    test_rows = [
        retag(row, dataset_name, 'test', i, (row.get('extra_info') or {}).get('index', str(i)))
        for i, row in enumerate(source_test)
    ]
    output_dir = Path(args.output_dir)
    write_dataset(output_dir, train_rows, test_rows)
    summary = {
        'source_dir': str(source_dir),
        'output_dir': str(output_dir),
        'selection_rule': 'deterministic random sample from source test split',
        'seed': args.seed,
        'pool_size': len(train_rows),
        'source_train_size': len(source_train),
        'source_test_size': len(source_test),
        'selected': selected,
    }
    write_json(output_dir / 'prepare_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def make_final(args):
    pool_dir = Path(args.pool_dir)
    source_test_dir = Path(args.test_source_dir)
    output_dir = Path(args.output_dir)
    pool_train = load_json(pool_dir / 'train.json')
    source_test = load_json(source_test_dir / 'test.json')
    by_index = {(row.get('extra_info') or {}).get('index'): row for row in pool_train}
    selected = []
    selected_records = []
    seen = set()
    for record in iter_jsonl(Path(args.scan_groups)):
        index = record.get('index') or (record.get('extra_info') or {}).get('index')
        if not index or index in seen or index not in by_index:
            continue
        majority = record.get('majority_prediction')
        truth = record.get('ground_truth')
        best_sample = None
        correct_sample_count = 0
        if args.selection_mode == 'best_of_n':
            correct_sample_count = sum(1 for sample in (record.get('samples') or []) if sample_is_correct(sample))
            best_sample = best_correct_sample(record)
            candidate_prediction = best_sample.get('prediction') if best_sample else None
            ok, details = is_match(candidate_prediction, truth, args.metric, args.smape_threshold)
            if best_sample is not None and not ok and float(best_sample.get('_reward_value', 0.0)) > 0.0:
                ok = True
                details = {
                    'normalized_prediction': normalize_direct_answer(candidate_prediction),
                    'normalized_truth': normalize_direct_answer(truth),
                    'matched_by_eval_reward': True,
                }
        else:
            candidate_prediction = majority
            ok, details = is_match(candidate_prediction, truth, args.metric, args.smape_threshold)
        if not ok:
            continue
        seen.add(index)
        src_index = (by_index[index].get('extra_info') or {}).get('source_index')
        selected.append(retag(by_index[index], output_dir.name, 'train', len(selected), src_index))
        selected_records.append({
            'pool_index': index,
            'source_index': src_index,
            'ground_truth': truth,
            'majority_prediction': majority,
            'majority_count': record.get('majority_count'),
            'majority_ratio': record.get('majority_ratio'),
            'prediction_counter': record.get('prediction_counter'),
            'selection_mode': args.selection_mode,
            'selected_by': args.selection_mode,
            'best_sample_index': best_sample.get('sample_index') if best_sample else None,
            'best_prediction': best_sample.get('prediction') if best_sample else None,
            'best_response_raw': best_sample.get('response_raw') if best_sample else None,
            'best_reward': best_sample.get('_reward_value') if best_sample else None,
            'correct_sample_count': correct_sample_count,
            **details,
        })
        if len(selected) >= args.final_size:
            break
    original_selected_count = len(selected)
    repeat_fill_records = []
    if len(selected) < args.final_size and args.allow_repeat_fill and selected_records:
        base_records = list(selected_records)
        while len(selected) < args.final_size:
            repeat_from = (len(selected) - original_selected_count) % original_selected_count
            base_record = base_records[repeat_from]
            pool_index = base_record['pool_index']
            src_index = base_record.get('source_index')
            row = retag(by_index[pool_index], output_dir.name, 'train', len(selected), src_index)
            extra = dict(row.get('extra_info') or {})
            extra['repeat_fill'] = True
            extra['repeat_from_selected_position'] = repeat_from
            row['extra_info'] = extra
            selected.append(row)
            repeat_record = dict(base_record)
            repeat_record['repeat_fill'] = True
            repeat_record['repeat_from_selected_position'] = repeat_from
            repeat_record['repeat_position'] = len(selected) - 1
            selected_records.append(repeat_record)
            repeat_fill_records.append(repeat_record)
    if len(selected) < args.final_size:
        raise RuntimeError(f'Only found {len(selected)} high-quality rows, need {args.final_size}.')
    test_rows = [
        retag(row, output_dir.name, 'test', i, (row.get('extra_info') or {}).get('index', str(i)))
        for i, row in enumerate(source_test)
    ]
    write_dataset(output_dir, selected, test_rows)
    summary = {
        'pool_dir': str(pool_dir),
        'scan_groups': str(args.scan_groups),
        'test_source_dir': str(source_test_dir),
        'output_dir': str(output_dir),
        'selection_rule': f'32-sample base {args.selection_mode} passes metric={args.metric}',
        'selection_mode': args.selection_mode,
        'metric': args.metric,
        'smape_threshold': args.smape_threshold,
        'final_size': len(selected),
        'unique_selected_size': original_selected_count,
        'repeat_fill_enabled': bool(args.allow_repeat_fill),
        'repeat_fill_size': len(repeat_fill_records),
        'test_size': len(test_rows),
        'selected': selected_records,
    }
    write_json(output_dir / 'prepare_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('make-pool')
    p.add_argument('--source-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--pool-size', type=int, default=80)
    p.add_argument('--seed', type=int, default=42)
    p.set_defaults(func=make_pool)
    p = sub.add_parser('make-final')
    p.add_argument('--pool-dir', required=True)
    p.add_argument('--scan-groups', required=True)
    p.add_argument('--test-source-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--final-size', type=int, default=20)
    p.add_argument('--metric', choices=['exact', 'numeric_smape', 'direct_answer'], default='exact')
    p.add_argument('--smape-threshold', type=float, default=0.9)
    p.add_argument('--selection-mode', choices=['majority', 'best_of_n'], default='majority')
    p.add_argument('--allow-repeat-fill', action='store_true')
    p.set_defaults(func=make_final)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
