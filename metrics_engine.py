from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional
import cv2
import numpy as np
from io_utils import load_image, resize_to_match, get_logger
log = get_logger(__name__)

@dataclass
class EvalResult:
    image_a: str
    image_b: str
    n_keypoints_a: int = 0
    n_keypoints_b: int = 0
    n_matches: int = 0
    n_inliers: int = 0
    inlier_ratio: float = 0.0
    mean_reproj_error: float = float('nan')
    color_hist_sim: float = float('nan')
    lpips_distance: float = float('nan')
    clip_similarity: float = float('nan')
    homography_ok: bool = False
    apparent_rotation_deg: float = float('nan')
    abs_apparent_rotation_deg: float = float('nan')
    apparent_scale: float = float('nan')
    apparent_tx: float = float('nan')
    apparent_ty: float = float('nan')
    note: str = ''

    def to_dict(self) -> dict:
        return asdict(self)

class SIFTMatcher:

    def __init__(self, n_features: int=0, contrast_threshold: float=0.04, edge_threshold: float=10, ratio_test: float=0.75):
        self.detector = cv2.SIFT_create(nfeatures=n_features, contrastThreshold=contrast_threshold, edgeThreshold=edge_threshold)
        self.ratio_test = ratio_test
        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

    def detect(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        kps, desc = self.detector.detectAndCompute(gray, None)
        return (kps, desc)

    def match(self, desc_a: np.ndarray, desc_b: np.ndarray) -> list:
        if desc_a is None or desc_b is None or len(desc_a) < 2 or (len(desc_b) < 2):
            return []
        try:
            knn = self.matcher.knnMatch(desc_a, desc_b, k=2)
        except cv2.error:
            return []
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.ratio_test * n.distance:
                good.append(m)
        return good

def estimate_homography(kps_a, kps_b, matches, threshold: float=5.0, confidence: float=0.99, max_iters: int=2000, min_matches: int=8) -> tuple[Optional[np.ndarray], np.ndarray]:
    if len(matches) < min_matches:
        return (None, np.array([], dtype=bool))
    src = np.float32([kps_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kps_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=max_iters, confidence=confidence)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), dtype=bool)
    return (H, inlier_mask)

def reprojection_error(kps_a, kps_b, matches, H, inlier_mask) -> float:
    if H is None or inlier_mask.sum() == 0:
        return float('nan')
    src = np.float32([kps_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kps_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src, H)
    errs = np.linalg.norm(projected - dst, axis=2).ravel()
    return float(errs[inlier_mask].mean())

def estimate_affine_proxy(kps_a, kps_b, matches, threshold: float=5.0, confidence: float=0.99, max_iters: int=2000, min_matches: int=3) -> tuple[Optional[np.ndarray], np.ndarray]:
    if len(matches) < min_matches:
        return (None, np.array([], dtype=bool))
    src = np.float32([kps_a[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kps_b[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=threshold, maxIters=max_iters, confidence=confidence)
    inlier_mask = mask.ravel().astype(bool) if mask is not None else np.zeros(len(matches), dtype=bool)
    return (M, inlier_mask)

def decompose_affine_proxy(M: np.ndarray) -> tuple[float, float, float, float]:
    a = float(M[0, 0])
    b = float(M[1, 0])
    tx = float(M[0, 2])
    ty = float(M[1, 2])
    scale = float(np.hypot(a, b))
    rotation_deg = float(np.degrees(np.arctan2(b, a)))
    return (rotation_deg, scale, tx, ty)

def color_histogram_similarity(img_a: np.ndarray, img_b: np.ndarray) -> float:
    hsv_a = cv2.cvtColor(img_a, cv2.COLOR_RGB2HSV)
    hsv_b = cv2.cvtColor(img_b, cv2.COLOR_RGB2HSV)
    bins = [50, 60]
    ranges = [0, 180, 0, 256]
    hist_a = cv2.calcHist([hsv_a], [0, 1], None, bins, ranges)
    hist_b = cv2.calcHist([hsv_b], [0, 1], None, bins, ranges)
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)
    return float(cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL))

class LPIPSMetric:

    def __init__(self, net: str='alex', device: str='cpu'):
        try:
            import torch
            import lpips
        except ImportError as e:
            raise ImportError(f'LPIPS requires `pip install torch lpips`. Original error: {e}')
        self.torch = torch
        self.device = device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device)
        self.model.eval()

    def _to_tensor(self, img: np.ndarray):
        t = self.torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
        t = t / 127.5 - 1.0
        return t.to(self.device)

    def distance(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        with self.torch.no_grad():
            d = self.model(self._to_tensor(img_a), self._to_tensor(img_b))
        return float(d.item())

class CLIPSimilarity:

    def __init__(self, model_name: str='ViT-B-32', pretrained: str='laion2b_s34b_b79k', device: str='cpu'):
        try:
            import torch
            import open_clip
            from PIL import Image
        except ImportError as e:
            raise ImportError(f'CLIP requires `pip install torch open_clip_torch pillow`. Original error: {e}')
        self.torch = torch
        self.Image = Image
        self.device = device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(device).eval()

    def _features(self, img: np.ndarray):
        pil = self.Image.fromarray(img)
        t = self.preprocess(pil).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            f = self.model.encode_image(t)
            f = f / f.norm(dim=-1, keepdim=True)
        return f

    def similarity(self, img_a: np.ndarray, img_b: np.ndarray) -> float:
        f_a = self._features(img_a)
        f_b = self._features(img_b)
        sim = (f_a @ f_b.T).item()
        return float(sim)

class ViewpointEvaluator:

    def __init__(self, config: dict):
        m = config['matcher']
        self.matcher = SIFTMatcher(n_features=m['sift']['n_features'], contrast_threshold=m['sift']['contrast_threshold'], edge_threshold=m['sift']['edge_threshold'], ratio_test=m['ratio_test'])
        self.ransac = config['ransac']
        self.lpips_model = None
        self.clip_model = None
        deep = config.get('deep_metrics', {})
        if deep.get('use_lpips', False):
            try:
                log.info('Loading LPIPS model...')
                self.lpips_model = LPIPSMetric(net=deep.get('lpips_net', 'alex'), device=deep.get('device', 'cpu'))
                log.info('  LPIPS loaded.')
            except Exception as e:
                log.warning(f'LPIPS unavailable, skipping: {e}')
                self.lpips_model = None
        if deep.get('use_clip', False):
            try:
                log.info('Loading CLIP model...')
                self.clip_model = CLIPSimilarity(model_name=deep.get('clip_model', 'ViT-B-32'), pretrained=deep.get('clip_pretrained', 'laion2b_s34b_b79k'), device=deep.get('device', 'cpu'))
                log.info('  CLIP loaded.')
            except Exception as e:
                log.warning(f'CLIP unavailable, skipping: {e}')
                self.clip_model = None

    def evaluate(self, path_a: str | Path, path_b: str | Path, resize_match: bool=True) -> EvalResult:
        result = EvalResult(image_a=str(path_a), image_b=str(path_b))
        try:
            img_a = load_image(path_a)
            img_b = load_image(path_b)
        except Exception as e:
            result.note = f'load_failed: {e}'
            return result
        if resize_match:
            img_a, img_b = resize_to_match(img_a, img_b)
        try:
            result.color_hist_sim = color_histogram_similarity(img_a, img_b)
        except Exception as e:
            log.warning(f'color_hist failed: {e}')
        if self.lpips_model is not None:
            try:
                result.lpips_distance = self.lpips_model.distance(img_a, img_b)
            except Exception as e:
                log.warning(f'LPIPS failed: {e}')
        if self.clip_model is not None:
            try:
                result.clip_similarity = self.clip_model.similarity(img_a, img_b)
            except Exception as e:
                log.warning(f'CLIP failed: {e}')
        kps_a, desc_a = self.matcher.detect(img_a)
        kps_b, desc_b = self.matcher.detect(img_b)
        result.n_keypoints_a = len(kps_a) if kps_a else 0
        result.n_keypoints_b = len(kps_b) if kps_b else 0
        if result.n_keypoints_a == 0 or result.n_keypoints_b == 0:
            result.note = 'no_keypoints'
            return result
        matches = self.matcher.match(desc_a, desc_b)
        result.n_matches = len(matches)
        if result.n_matches < self.ransac['min_matches']:
            result.note = f'insufficient_matches ({result.n_matches})'
            return result
        H, inlier_mask = estimate_homography(kps_a, kps_b, matches, threshold=self.ransac['threshold'], confidence=self.ransac['confidence'], max_iters=self.ransac['max_iters'], min_matches=self.ransac['min_matches'])
        result.n_inliers = int(inlier_mask.sum())
        result.inlier_ratio = result.n_inliers / result.n_matches if result.n_matches > 0 else 0.0
        result.homography_ok = H is not None and result.n_inliers >= 4
        if result.homography_ok:
            result.mean_reproj_error = reprojection_error(kps_a, kps_b, matches, H, inlier_mask)
        M, _ = estimate_affine_proxy(kps_a, kps_b, matches, threshold=self.ransac['threshold'], confidence=self.ransac['confidence'], max_iters=self.ransac['max_iters'])
        if M is not None:
            result.apparent_rotation_deg, result.apparent_scale, result.apparent_tx, result.apparent_ty = decompose_affine_proxy(M)
            result.abs_apparent_rotation_deg = abs(result.apparent_rotation_deg)
        return result

def sanity_check(evaluator: ViewpointEvaluator, image_path: str | Path, expected_inlier_ratio: float=0.95) -> bool:
    result = evaluator.evaluate(image_path, image_path)
    log.info(f'Sanity check on {image_path}:')
    log.info(f'  n_matches={result.n_matches}, n_inliers={result.n_inliers}, inlier_ratio={result.inlier_ratio:.3f}, reproj_err={result.mean_reproj_error:.3f}')
    if not result.lpips_distance != result.lpips_distance:
        log.info(f'  lpips={result.lpips_distance:.4f} (expected ~0)')
    if not result.clip_similarity != result.clip_similarity:
        log.info(f'  clip_sim={result.clip_similarity:.4f} (expected ~1)')
    ok = result.inlier_ratio >= expected_inlier_ratio
    if not ok:
        log.warning(f'Sanity check FAILED: inlier_ratio={result.inlier_ratio:.3f} < expected {expected_inlier_ratio}')
    else:
        log.info('Sanity check PASSED.')
    return ok
