import argparse
import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm
sys.path.insert(0, str(Path(__file__).parent))
from io_utils import load_config, get_logger, load_metadata, ensure_dir
from image_registry import import_generated_directory, register_input_directory
from metrics_engine import ViewpointEvaluator, sanity_check
log = get_logger('eval')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'stage3'], help='Evaluate only generated images from a single stage.')
    parser.add_argument('--output', help='Optional output CSV path. Defaults to results/metrics.csv, or results/metrics_<stage>.csv when --stage is used.')
    parser.add_argument('--compare-to-buckets', nargs='+', choices=['small', 'medium', 'large', 'total', 'stage3'], help='Compare selected generated images against matching generated images from these folder buckets instead of the original input.')
    parser.add_argument('--default-reference-bucket', choices=['small', 'medium', 'large', 'total', 'stage3'], help='Default generated reference bucket to use for every evaluated image.')
    parser.add_argument('--reference-override', action='append', default=[], help='Override generated reference bucket for a specific image_id. Format: image_id=bucket')
    return parser.parse_args()

def parse_reference_overrides(raw_overrides: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    valid_buckets = {'small', 'medium', 'large', 'total', 'stage3'}
    for item in raw_overrides:
        if '=' not in item:
            raise ValueError(f"Invalid --reference-override '{item}'. Expected image_id=bucket.")
        image_id, bucket = item.split('=', 1)
        image_id = image_id.strip()
        bucket = bucket.strip()
        if bucket not in valid_buckets:
            raise ValueError(f"Invalid bucket '{bucket}' in --reference-override '{item}'.")
        overrides[image_id] = bucket
    return overrides

def main():
    args = parse_args()
    overrides = parse_reference_overrides(args.reference_override)
    config = load_config('config.yaml')
    paths = config['paths']
    metadata_path = paths['metadata_file']
    log.info(f'Working directory: {Path.cwd()}')
    log.info(f"Generated dir: {Path(paths['generated_dir']).resolve()}")
    default_out = Path(paths['results_dir']) / (f'metrics_{args.stage}.csv' if args.stage else 'metrics.csv')
    out_path = Path(args.output) if args.output else default_out
    log.info(f'Metrics output: {out_path.resolve()}')
    md_path = Path(metadata_path)
    if md_path.exists():
        md_path.unlink()
    lookup = register_input_directory(paths['inputs_dir'], metadata_path)
    if not lookup:
        log.error('No input images. Run setup_experiment.py first.')
        return
    n = import_generated_directory(paths['generated_dir'], metadata_path, lookup)
    if n == 0:
        log.error('No generated images found. Place them in data/generated/ first.')
        return
    df = load_metadata(metadata_path)
    log.info(f"Total registered images: {len(df)} (inputs={sum(df['role'] == 'input')}, generated={sum(df['role'] == 'generated')})")
    evaluator = ViewpointEvaluator(config)
    inputs = df[df['role'] == 'input']
    if len(inputs) > 0:
        first_input = inputs.iloc[0]['file_path']
        log.info('Running sanity check...')
        ok = sanity_check(evaluator, first_input, config['sanity']['expected_inlier_ratio_self'])
        if not ok:
            log.warning('Sanity check failed - proceeding anyway, but verify pipeline')
    generated = df[df['role'] == 'generated']
    if args.stage:
        generated = generated[generated['stage'] == args.stage]
        log.info(f'Filtering to stage={args.stage}: {len(generated)} generated images')
    if len(generated) == 0:
        log.error('No generated images to evaluate.')
        return
    input_path_lookup = dict(zip(inputs['image_id'], inputs['file_path']))
    ref_lookup = {}
    compare_mode = 'input'
    if args.compare_to_buckets:
        compare_mode = 'generated'
        ref_generated = df[(df['role'] == 'generated') & df['generated_bucket'].isin(args.compare_to_buckets)].copy()
        ref_lookup = {(row['source_input_id'], row['generated_bucket']): row for _, row in ref_generated.iterrows()}
        log.info('Comparing against generated references from buckets=%s (%d reference images)', args.compare_to_buckets, len(ref_generated))
    elif args.default_reference_bucket or overrides:
        compare_mode = 'generated'
        requested_buckets = sorted({args.default_reference_bucket, *overrides.values()} - {None})
        ref_generated = df[(df['role'] == 'generated') & df['generated_bucket'].isin(requested_buckets)].copy()
        ref_lookup = {(row['source_input_id'], row['generated_bucket']): row for _, row in ref_generated.iterrows()}
        log.info('Comparing against generated references with default bucket=%s, overrides=%s', args.default_reference_bucket, overrides or '{}')
    rows = []
    for _, gen_row in tqdm(generated.iterrows(), total=len(generated), desc='Evaluating'):
        src_id = gen_row['source_input_id']
        gen_path = gen_row['file_path']
        if compare_mode == 'generated':
            if args.compare_to_buckets:
                ref_buckets = args.compare_to_buckets
            else:
                chosen_bucket = overrides.get(gen_row['image_id'], args.default_reference_bucket)
                if not chosen_bucket:
                    log.warning('No generated reference bucket specified for %s', gen_row['image_id'])
                    continue
                ref_buckets = [chosen_bucket]
        else:
            ref_buckets = [None]
        for ref_bucket in ref_buckets:
            if compare_mode == 'generated':
                ref_row = ref_lookup.get((src_id, ref_bucket))
                if ref_row is None:
                    log.warning('No generated reference found for %s in bucket=%s', gen_row['image_id'], ref_bucket)
                    continue
                ref_path = ref_row['file_path']
                ref_image_id = ref_row['image_id']
            else:
                if src_id not in input_path_lookup:
                    log.warning(f"No input found for {gen_row['image_id']} (src={src_id})")
                    continue
                ref_path = input_path_lookup[src_id]
                ref_image_id = src_id
            result = evaluator.evaluate(ref_path, gen_path)
            rows.append({'image_id': gen_row['image_id'], 'category': gen_row['category'], 'stage': gen_row['stage'], 'generated_bucket': gen_row.get('generated_bucket', ''), 'magnitude': gen_row['magnitude'], 'condition': gen_row['condition'], 'source_input_id': src_id, 'comparison_mode': compare_mode, 'reference_bucket': ref_bucket or 'input', 'reference_image_id': ref_image_id, 'n_keypoints_a': result.n_keypoints_a, 'n_keypoints_b': result.n_keypoints_b, 'n_matches': result.n_matches, 'n_inliers': result.n_inliers, 'inlier_ratio': result.inlier_ratio, 'mean_reproj_error': result.mean_reproj_error, 'color_hist_sim': result.color_hist_sim, 'lpips_distance': result.lpips_distance, 'clip_similarity': result.clip_similarity, 'apparent_rotation_deg': result.apparent_rotation_deg, 'abs_apparent_rotation_deg': result.abs_apparent_rotation_deg, 'apparent_scale': result.apparent_scale, 'apparent_tx': result.apparent_tx, 'apparent_ty': result.apparent_ty, 'homography_ok': result.homography_ok, 'note': result.note})
    out_df = pd.DataFrame(rows)
    ensure_dir(out_path.parent)
    out_df.to_csv(out_path, index=False)
    log.info(f'Saved metrics: {out_path}')
    log.info(f'Evaluated {len(out_df)} generated images.')
    if len(out_df):
        log.info('Metrics rows by generated folder:')
        log.info(f"\n{out_df['generated_bucket'].value_counts(dropna=False)}")
        if 'reference_bucket' in out_df.columns:
            log.info('Metrics rows by reference bucket:')
            log.info(f"\n{out_df['reference_bucket'].value_counts(dropna=False)}")
    log.info('\n=== Quick Summary ===')
    metric_cols = ['inlier_ratio', 'color_hist_sim', 'lpips_distance', 'clip_similarity', 'n_matches', 'apparent_rotation_deg', 'abs_apparent_rotation_deg', 'apparent_scale']
    metric_cols = [c for c in metric_cols if c in out_df.columns and (not out_df[c].isna().all())]
    stage1_order = ['small', 'medium', 'large']
    s1 = out_df[out_df['generated_bucket'].isin(stage1_order)].copy()
    if len(s1):
        s1['generated_bucket'] = pd.Categorical(s1['generated_bucket'], categories=stage1_order, ordered=True)
        log.info('\nStage 1 - by generated folder:')
        log.info(f"\n{s1.groupby('generated_bucket', observed=False)[metric_cols].mean().round(4)}")
    s2 = out_df[out_df['generated_bucket'] == 'total']
    if len(s2):
        log.info('\nStage 2 - total folder by category:')
        log.info(f"\n{s2.groupby('category')[metric_cols].mean().round(4)}")
    if 'reference_bucket' in out_df.columns and compare_mode == 'generated':
        log.info('\nComparison by reference bucket:')
        log.info(f"\n{out_df.groupby('reference_bucket')[metric_cols].mean().round(4)}")
if __name__ == '__main__':
    main()
