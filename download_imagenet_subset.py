from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Optional
import yaml

def load_imagenet_class_mapping(config_path: Path) -> dict[str, list[tuple[int, str]]]:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    raw = cfg.get('imagenet_classes')
    if not raw:
        raise ValueError("config.yaml is missing the 'imagenet_classes' section. Expected: imagenet_classes: {tier: [[idx, name], ...]}")
    mapping = {}
    for tier, entries in raw.items():
        mapping[tier] = [(int(idx), str(name)) for idx, name in entries]
    return mapping

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('config.yaml'))
    p.add_argument('--out-dir', type=Path, default=Path('data/inputs'), help='Output root; images go to {out-dir}/{tier}/')
    p.add_argument('--split', type=str, default='validation', choices=['validation', 'train'], help='ImageNet split to stream from. Validation is smaller (50K imgs, 50 per class) and faster.')
    p.add_argument('--n-per-class', type=int, default=5, help='Number of images to save per ImageNet class. Default 5 lets you visually pick the best one afterward.')
    p.add_argument('--clear', action='store_true', help="Delete existing tier folders' contents first.")
    args = p.parse_args()
    try:
        from datasets import load_dataset
        from PIL import Image
    except ImportError as e:
        print('ERROR: missing dependency. Install with:', file=sys.stderr)
        print('  pip install datasets pillow', file=sys.stderr)
        print(f'  (original: {e})', file=sys.stderr)
        sys.exit(1)
    mapping = load_imagenet_class_mapping(args.config)
    print('=' * 60)
    print('ImageNet target classes')
    print('=' * 60)
    for tier, entries in mapping.items():
        print(f'  [{tier}] {len(entries)} classes:')
        for idx, name in entries:
            print(f'    {idx:4d}  {name}')
    total_classes = sum((len(v) for v in mapping.values()))
    target_total = total_classes * args.n_per_class
    print(f'\nWill save up to {target_total} images ({total_classes} classes x {args.n_per_class} each)')
    print()
    target_lookup: dict[int, tuple[str, str]] = {}
    for tier, entries in mapping.items():
        for idx, name in entries:
            target_lookup[idx] = (tier, name)
    saved_per_class: dict[int, int] = {idx: 0 for idx in target_lookup}
    needed = lambda idx: saved_per_class[idx] < args.n_per_class
    all_done = lambda: all((c >= args.n_per_class for c in saved_per_class.values()))
    next_idx_per_tier: dict[str, int] = {t: 1 for t in mapping}
    if args.clear:
        for tier in mapping:
            tier_dir = args.out_dir / tier
            if tier_dir.is_dir():
                for f in tier_dir.glob(f'{tier}_*.png'):
                    f.unlink()
    print(f'Streaming ILSVRC/imagenet-1k split={args.split} ...')
    print('(This may take a few minutes; progress prints below.)')
    try:
        ds = load_dataset('ILSVRC/imagenet-1k', split=args.split, streaming=True)
    except Exception as e:
        print(f'\nERROR: failed to load dataset: {e}', file=sys.stderr)
        print('\nMost likely causes:', file=sys.stderr)
        print("  1. You haven't agreed to the ImageNet terms on", file=sys.stderr)
        print('     https://huggingface.co/datasets/ILSVRC/imagenet-1k', file=sys.stderr)
        print("  2. You haven't run `huggingface-cli login` with a Read token.", file=sys.stderr)
        sys.exit(1)
    n_seen = 0
    n_saved = 0
    for item in ds:
        n_seen += 1
        if n_seen % 1000 == 0:
            print(f'  [progress] scanned {n_seen} images, saved {n_saved}/{target_total}')
        cls_idx = item['label']
        if cls_idx not in target_lookup or not needed(cls_idx):
            continue
        tier, short_name = target_lookup[cls_idx]
        idx_in_tier = next_idx_per_tier[tier]
        out_dir = args.out_dir / tier
        out_dir.mkdir(parents=True, exist_ok=True)
        copy_num = saved_per_class[cls_idx] + 1
        if copy_num == 1:
            fname = f'{tier}_{idx_in_tier:02d}_{short_name}.png'
        else:
            fname = f'{tier}_{idx_in_tier:02d}_{short_name}-{copy_num}.png'
        out_path = out_dir / fname
        try:
            img = item['image']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(out_path, 'PNG')
            saved_per_class[cls_idx] += 1
            if copy_num == 1:
                next_idx_per_tier[tier] += 1
            n_saved += 1
            print(f'  saved {fname}  (class {cls_idx}, copy {copy_num})')
        except Exception as e:
            print(f'  WARNING: failed to save image for class {cls_idx}: {e}', file=sys.stderr)
        if all_done():
            print('\nAll target classes saturated. Stopping stream.')
            break
    print()
    print('=' * 60)
    print('Done.')
    print('=' * 60)
    for idx, count in saved_per_class.items():
        tier, name = target_lookup[idx]
        flag = 'OK ' if count >= args.n_per_class else 'MISSING'
        print(f'  [{flag}] {tier:12s} {idx:4d} {name:20s} {count}/{args.n_per_class}')
    print()
    print(f'Saved {n_saved} images out of {target_total} target.')
    print(f'Output: {args.out_dir.resolve()}')
    print()
    print('Next steps:')
    print('  1. Visually review the saved images. ImageNet sample quality')
    print("     varies; replace any that aren't good frontal/three-quarter")
    print('     views with another from the same class folder, or rerun')
    print('     this script with --n-per-class 10 for more options.')
    print("  2. Once you've kept the best one per class, rename them so")
    print('     the filenames follow {tier}_{idx:02d}_{name}.png with')
    print('     idx going 01-05 in each tier.')
    print('  3. Run:  python setup_experiment.py')
if __name__ == '__main__':
    main()
