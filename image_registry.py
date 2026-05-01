from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
from io_utils import ImageMeta, append_metadata, get_logger, ensure_dir
log = get_logger(__name__)
PROMPT_TEMPLATES = {'stage1': {'small': 'Show me this same object from a slightly rotated angle, as if the camera moved a little to the right (approximately 15-20 degrees). Keep all the parts, colors, and proportions exactly the same as in the original.', 'medium': 'Show me this same object from a clear side angle, viewed from the right side (approximately 45 degrees rotation). Keep all the parts, colors, and proportions exactly the same as in the original.', 'large': 'Show me this same object from behind, as if the camera moved to the back of the object (approximately 90-180 degrees rotation). Keep all the parts, colors, and proportions exactly the same as in the original.'}, 'stage2': {'medium': 'Show me this same object from a clear side angle, viewed from the right side. Keep all the parts, colors, and proportions exactly the same as in the original.'}, 'stage3': {'baseline': 'Show me this same object from a side angle (approximately 45 degrees).', 'viewpoint_desc': "Imagine this object placed in 3D space with its position and orientation fixed. Now rotate the camera 45 degrees clockwise around the vertical axis passing through the object's center, keeping the same camera height and the same distance from the object. Render this new view, preserving the object's 3D structural relationships, occlusion patterns between its parts, and all colors and proportions."}}

def get_prompt(stage: str, key: str) -> str:
    return PROMPT_TEMPLATES[stage][key]
FILENAME_RE = re.compile('^(?P<input_id>[a-z0-9_-]+)__(?P<stage>stage[1-3])__(?P<key>[a-z_]+)\\.(png|jpg|jpeg)$', re.IGNORECASE)

def parse_generated_filename(name: str) -> Optional[dict]:
    m = FILENAME_RE.match(name)
    if not m:
        stem = Path(name).stem
        input_id = stem.split('__', 1)[0]
        if not input_id:
            return None
        return {'input_id': input_id, 'stage': '', 'key': ''}
    return m.groupdict()

def infer_generated_attrs(bucket: str, parsed: dict) -> dict:
    bucket = bucket.lower()
    stage = parsed.get('stage', '')
    key = parsed.get('key', '')
    if bucket in {'small', 'medium', 'large'}:
        return {'stage': 'stage1', 'magnitude': bucket, 'condition': '', 'generated_bucket': bucket}
    if bucket == 'total':
        return {'stage': 'stage2' if not stage else stage, 'magnitude': key if key in {'small', 'medium', 'large'} else '', 'condition': key if key in {'baseline', 'viewpoint_desc'} else '', 'generated_bucket': bucket}
    if bucket == 'stage3':
        return {'stage': 'stage3', 'magnitude': '', 'condition': key if key else bucket, 'generated_bucket': bucket}
    return {'stage': stage or bucket or 'generated', 'magnitude': key if key in {'small', 'medium', 'large'} else '', 'condition': key if key in {'baseline', 'viewpoint_desc'} else '', 'generated_bucket': bucket}

def import_generated_directory(generated_dir: str | Path, metadata_path: str | Path, category_lookup: dict) -> int:
    gen_dir = Path(generated_dir)
    if not gen_dir.exists():
        log.warning(f'Generated dir does not exist: {gen_dir}')
        return 0
    n_imported = 0
    bucket_counts: dict[str, int] = {}
    for f in sorted(gen_dir.rglob('*')):
        if not f.is_file():
            continue
        if f.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
            continue
        parsed = parse_generated_filename(f.name)
        if parsed is None:
            log.warning(f'Skipping unrecognized filename: {f.name}')
            continue
        input_id = parsed['input_id']
        bucket = f.relative_to(gen_dir).parts[0] if f.parent != gen_dir else ''
        attrs = infer_generated_attrs(bucket, parsed)
        stage = attrs['stage']
        key = parsed['key']
        category = category_lookup.get(input_id, 'unknown')
        magnitude = attrs['magnitude']
        condition = attrs['condition']
        prompt = ''
        try:
            prompt = get_prompt(stage, key)
        except KeyError:
            pass
        meta = ImageMeta(image_id=str(f.relative_to(gen_dir).with_suffix('')).replace('/', '__'), role='generated', category=category, stage=stage, generated_bucket=attrs['generated_bucket'], magnitude=magnitude, condition=condition, source_input_id=input_id, prompt=prompt, file_path=str(f.resolve()))
        append_metadata(meta, metadata_path)
        n_imported += 1
        bucket_name = attrs['generated_bucket'] or '(root)'
        bucket_counts[bucket_name] = bucket_counts.get(bucket_name, 0) + 1
    log.info(f'Imported {n_imported} generated images.')
    if bucket_counts:
        log.info(f'Generated image counts by folder: {bucket_counts}')
    return n_imported

def register_input_directory(inputs_dir: str | Path, metadata_path: str | Path) -> dict:
    inputs_dir = Path(inputs_dir)
    lookup = {}
    for cat_dir in sorted(inputs_dir.iterdir()):
        if not cat_dir.is_dir():
            continue
        category = cat_dir.name
        for f in sorted(cat_dir.iterdir()):
            if f.suffix.lower() not in {'.png', '.jpg', '.jpeg'}:
                continue
            input_id = f.stem
            lookup[input_id] = category
            meta = ImageMeta(image_id=input_id, role='input', category=category, stage='input', file_path=str(f.resolve()))
            append_metadata(meta, metadata_path)
    log.info(f'Registered {len(lookup)} input images.')
    return lookup

class APIGenerator:

    def __init__(self, api_key: Optional[str]=None):
        self.api_key = api_key
        log.warning('APIGenerator is a stub. Use ManualLoader for now.')

    def generate(self, input_image_path: str, prompt: str, out_path: str) -> str:
        raise NotImplementedError('API generation not implemented. Use manual collection via ChatGPT web UI.')
