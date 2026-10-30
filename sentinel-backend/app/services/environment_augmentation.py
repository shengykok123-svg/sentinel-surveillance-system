"""Deterministic simulated camera degradations for offline enrollment experiments.

No restoration, identity synthesis, training, or production configuration changes.
"""
import hashlib
import cv2
import numpy as np


def stable_seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], 'little')


def simulate_environment(image, *, exposure=1., noise_sigma=0., scale=1., seed=0):
    """BGR uint8 -> same size BGR uint8. Scale simulates reduced pixel sampling.

    Exposure multiplication is an image-domain approximation, not a calibrated
    sensor or physically accurate low-light renderer. Noise is additive Gaussian.
    """
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected BGR uint8 image')
    if not (0 < scale <= 1 and 0 < exposure <= 1 and noise_sigma >= 0):
        raise ValueError('Invalid degradation parameters')
    h,w=image.shape[:2]
    result=image.copy()
    if scale < 1:
        small=cv2.resize(result,(max(2,round(w*scale)),max(2,round(h*scale))),interpolation=cv2.INTER_AREA)
        result=cv2.resize(small,(w,h),interpolation=cv2.INTER_LINEAR)
    if exposure != 1 or noise_sigma:
        result=result.astype(np.float32)*exposure
        if noise_sigma:
            rng=np.random.default_rng(seed)
            result += rng.standard_normal(result.shape,dtype=np.float32)*noise_sigma
        result=np.clip(np.rint(result),0,255).astype(np.uint8)
    return result


def select_balanced_candidates(candidates, budget=4):
    """Round-robin Neutral/Smile and yaw buckets; no probe scores are consulted.

    Prefer valid +/-30 views, then real originals, +/-15 and +/-45. The fixed
    ordering tests whether degradation of posed faces helps, rather than selecting
    only the easiest frontal candidates. One view per source key.
    """
    priorities={-30:0,30:1,0:2,-15:3,15:4,-45:5,45:6}
    result=[]
    ordered=sorted(candidates,key=lambda r:(priorities[int(r['yaw'])],r['expression'],r['source_path']))
    for row in ordered:
        if len(result)>=budget:break
        result.append(row)
    return result
