import argparse
import glob
import os
from typing import List, Optional, Tuple
import csv

import cv2
import numpy as np


def _build_color_mask(color_img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 120, 50), (180, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    mask = cv2.dilate(mask, kernel)
    return mask


def _create_detector(name: str):
    name = name.lower()
    if name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            return None
        return cv2.SIFT_create()
    if name == "akaze":
        return cv2.AKAZE_create()
    if name == "orb":
        return cv2.ORB_create(nfeatures=6000)
    return None


def _detect_and_compute(
    detector, image_gray: np.ndarray, mask: Optional[np.ndarray] = None
):
    kps, des = detector.detectAndCompute(image_gray, mask)
    if des is None or len(kps) == 0:
        return [], None
    return kps, des


def _match_descriptors(detector_name: str, des1, des2):
    if detector_name in {"sift"}:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    return good


def _estimate_homography(
    kp1, kp2, matches, ransac_thresh: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if len(matches) < 4:
        return None, None
    src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)
    return H, mask


def _try_detectors(
    template_gray: np.ndarray,
    image_gray: np.ndarray,
    template_color: np.ndarray,
    image_color: np.ndarray,
    detector_names: List[str],
    ransac_thresh: float,
    debug_dir: Optional[str] = None,
    image_name: str = "",
):
    mask_t = _build_color_mask(template_color)
    mask_i = _build_color_mask(image_color)

    # ---- debug: save masks ----
    if debug_dir:
        cv2.imwrite(os.path.join(debug_dir, "mask_template.png"), mask_t)
        cv2.imwrite(os.path.join(debug_dir, f"mask_{image_name}"), mask_i)

    best = None
    best_inliers = -1
    best_viz = None

    for name in detector_names:
        detector = _create_detector(name)
        if detector is None:
            continue
        kp_t, des_t = _detect_and_compute(detector, template_gray, mask_t)
        kp_i, des_i = _detect_and_compute(detector, image_gray, mask_i)

        print(f"  [{name}] kp_template={len(kp_t)}  kp_image={len(kp_i)}", end="")

        if des_t is None or des_i is None:
            print(" -> skip (no descriptors)")
            continue

        matches = _match_descriptors(name, des_t, des_i)
        H, mask = _estimate_homography(kp_t, kp_i, matches, ransac_thresh)

        if H is None or mask is None:
            print(f" -> matches={len(matches)} -> RANSAC failed")
            continue

        inliers = int(mask.sum())
        print(f" -> matches={len(matches)}  inliers={inliers}")

        if inliers > best_inliers:
            best_inliers = inliers
            best = (H, name, inliers, len(matches))

            # ---- debug: save match visualization ----
            if debug_dir:
                match_mask = mask.ravel().tolist()
                viz = cv2.drawMatches(
                    template_color, kp_t,
                    image_color, kp_i,
                    matches, None,
                    matchColor=(0, 255, 0),
                    singlePointColor=(255, 0, 0),
                    matchesMask=match_mask,
                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
                )
                scale = 1200 / max(viz.shape[1], 1)
                if scale < 1:
                    viz = cv2.resize(viz, None, fx=scale, fy=scale)
                cv2.imwrite(
                    os.path.join(debug_dir, f"match_{name}_{image_name}"), viz
                )

    return best


def restore_images(
    data_dir: str,
    template_path: str,
    out_dir: str,
    method: str,
    ransac_thresh: float,
    debug: bool = False,
) -> None:
    stats = [] # (fname, detector, inliers, total)
    os.makedirs(out_dir, exist_ok=True)
    debug_dir = os.path.join(out_dir, "_debug") if debug else None
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)

    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template.shape[:2]

    image_paths = sorted(
        glob.glob(os.path.join(data_dir, "raw_*_warp*.png"))
    )
    if not image_paths:
        raise FileNotFoundError(f"No input images in {data_dir}")

    if method == "auto":
        detector_names = ["sift", "akaze", "orb"]
    else:
        detector_names = [method]

    for path in image_paths:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read {path}")
            continue
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        fname = os.path.basename(path)

        print(f"\n--- {fname} ---")
        result = _try_detectors(
            template_gray, img_gray,
            template, img,
            detector_names, ransac_thresh,
            debug_dir=debug_dir,
            image_name=fname,
        )
        if result is None:
            print(f"[WARN] Homography failed: {fname}")
            stats.append((fname, "FAILED", 0, 0))
            continue

        H, name, inliers, total = result
        warped = cv2.warpPerspective(img, H, (w, h), flags=cv2.INTER_LINEAR)

        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, warped)
        print(f"[OK] {fname} | {name} inliers {inliers}/{total}")
        stats.append((fname, name, inliers, total))
    
    csv_path = os.path.join(out_dir, "stats.csv")
    statistics_report(stats, csv_path=csv_path)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Restore homography to top view using keypoint matching."
    )
    p.add_argument("--data-dir", default="../HW1_data")
    p.add_argument("--template", default="../HW1_data/template.png")
    p.add_argument("--out-dir", default="../restored_images")
    p.add_argument(
        "--method", choices=["auto", "orb", "akaze", "sift"], default="auto"
    )
    p.add_argument("--ransac-thresh", type=float, default=5.0)
    p.add_argument(
        "--debug",
        action="store_true",
        help="Save mask/keypoint/match debug images to <out-dir>/_debug/",
    )
    return p

def statistics_report(stats, csv_path: Optional[str] = None):
    if stats:
        print("\n" + "=" * 60)
        print(f"{'File':<25} {'Detector':<8} {'Inliers':>8} {'Total':>7} {'Ratio':>7}")
        print("-" * 60)
        for fname, det, inl, tot in stats:
            ratio = f"{inl/tot:.1%}" if tot > 0 else "N/A"
            print(f"{fname:<25} {det:<8} {inl:>8} {tot:>7} {ratio:>7}")
        print("=" * 60)

        success = [(f, d, i, t) for f, d, i, t in stats if d != "FAILED"]
        failed  = [f for f, d, i, t in stats if d == "FAILED"]
        if success:
            inlier_counts = [i for _, _, i, _ in success]
            ratios = [i/t for _, _, i, t in success if t > 0]
            print(f"\nTotal images : {len(stats)}")
            print(f"Success      : {len(success)}  |  Failed: {len(failed)}")
            # print(f"Inliers  min={min(inlier_counts)}  max={max(inlier_counts)}  "
            #       f"mean={sum(inlier_counts)/len(inlier_counts):.1f}")
            # print(f"Ratio    min={min(ratios):.1%}  max={max(ratios):.1%}  "
            #       f"mean={sum(ratios)/len(ratios):.1%}")
            detector_used = {}
            for _, d, _, _ in success:
                detector_used[d] = detector_used.get(d, 0) + 1
            print(f"Detector breakdown: {detector_used}")
        if failed:
            print(f"Failed files : {failed}")
        
        # export CSV
        if csv_path:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "detector", "inliers", "total", "ratio"])
                for fname, det, inl, tot in stats:
                    ratio = f"{inl/tot:.4f}" if tot > 0 else ""
                    writer.writerow([fname, det, inl, tot, ratio])
            print(f"\nCSV saved to: {csv_path}")            


def main():
    args = build_argparser().parse_args()
    restore_images(
        data_dir=args.data_dir,
        template_path=args.template,
        out_dir=args.out_dir,
        method=args.method,
        ransac_thresh=args.ransac_thresh,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()