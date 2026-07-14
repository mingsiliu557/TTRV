#!/usr/bin/env python3
"""Summarize selection-only validation group JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def pct(value: float) -> float:
    return round(100.0 * value, 6)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', required=True, type=Path)
    parser.add_argument('--summary-json', required=True, type=Path)
    parser.add_argument('--summary-md', type=Path)
    parser.add_argument('--dataset', default='')
    args = parser.parse_args()

    groups = list(read_jsonl(args.groups)) if args.groups.exists() else []
    n = len(groups)
    if n == 0:
        summary = {'dataset': args.dataset, 'groups': 0, 'error': 'no groups found'}
    else:
        majority_correct = sum(bool(g.get('majority_correct')) for g in groups)
        reward_correct = sum(bool(g.get('reward_selection_correct')) for g in groups)
        pass_at_n = sum(bool(g.get('pass_at_n')) for g in groups)
        single_correct = sum(bool(g.get('single_correct')) for g in groups)
        agreement = sum(bool(g.get('major_reward_agreement')) for g in groups)
        reward_beats_major = sum(
            bool(g.get('reward_selection_correct')) and not bool(g.get('majority_correct')) for g in groups
        )
        major_beats_reward = sum(
            bool(g.get('majority_correct')) and not bool(g.get('reward_selection_correct')) for g in groups
        )
        invalid_total = sum(int(g.get('invalid_count') or 0) for g in groups)
        sample_total = sum(len(g.get('samples') or []) for g in groups)
        tie_groups = sum(int(g.get('reward_selection_tie_count') or 0) > 1 for g in groups)
        margins = [float(g.get('reward_selection_margin') or 0.0) for g in groups]
        summary = {
            'dataset': args.dataset,
            'groups': n,
            'samples': sample_total,
            'single_at_1': single_correct / n,
            'major_vote_at_32': majority_correct / n,
            'reward_select_at_32': reward_correct / n,
            'pass_at_32': pass_at_n / n,
            'major_reward_agreement': agreement / n,
            'reward_beats_major_count': reward_beats_major,
            'major_beats_reward_count': major_beats_reward,
            'invalid_ratio': invalid_total / sample_total if sample_total else 0.0,
            'reward_tie_rate': tie_groups / n,
            'reward_selection_margin_mean': sum(margins) / len(margins) if margins else 0.0,
        }
        summary.update({f'{key}_pct': pct(value) for key, value in list(summary.items()) if isinstance(value, float) and key.endswith(('at_1', 'at_32', 'agreement', 'ratio', 'rate'))})

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if args.summary_md:
        lines = [
            f"# Selection Summary: {args.dataset or args.groups.parent.name}",
            '',
            f"- groups: {summary.get('groups', 0)}",
        ]
        if summary.get('groups'):
            lines.extend([
                f"- single@1: {summary['single_at_1_pct']:.4f}",
                f"- major_vote@32: {summary['major_vote_at_32_pct']:.4f}",
                f"- reward_select@32: {summary['reward_select_at_32_pct']:.4f}",
                f"- pass@32: {summary['pass_at_32_pct']:.4f}",
                f"- agreement: {summary['major_reward_agreement_pct']:.4f}",
                f"- reward_beats_major_count: {summary['reward_beats_major_count']}",
                f"- major_beats_reward_count: {summary['major_beats_reward_count']}",
                f"- invalid_ratio: {summary['invalid_ratio_pct']:.4f}",
            ])
        args.summary_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
