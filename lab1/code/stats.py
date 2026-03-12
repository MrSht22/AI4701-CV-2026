import argparse
import csv
import os
import re
from collections import defaultdict
from statistics import mean, pstdev
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


FILE_PATTERN = re.compile(r"raw_(\d+)_warp(\d+)\.png$")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyze homography stats.csv by raw_0X groups and draw plots."
	)
	parser.add_argument(
		"--csv-path",
		default="../restored_images/stats.csv",
		help="Path to stats.csv exported by restore_homography.py",
	)
	parser.add_argument(
		"--out-dir",
		default="../restored_images/analysis",
		help="Directory to save analysis csv files and figures",
	)
	return parser.parse_args()


def load_records(csv_path: str) -> List[Dict]:
	records: List[Dict] = []
	with open(csv_path, "r", encoding="utf-8") as f:
		reader = csv.DictReader(f)
		for row in reader:
			file_name = row["file"].strip()
			m = FILE_PATTERN.match(file_name)
			if not m:
				continue

			raw_idx = int(m.group(1))
			warp_idx = int(m.group(2))
			inliers = int(row["inliers"])
			total = int(row["total"])
			ratio = float(row["ratio"]) if row["ratio"] else (inliers / total if total > 0 else 0.0)

			records.append(
				{
					"file": file_name,
					"raw_idx": raw_idx,
					"warp_idx": warp_idx,
					"inliers": inliers,
					"total": total,
					"ratio": ratio,
					"detector": row.get("detector", ""),
				}
			)
	return records


def summarize_by_group(records: List[Dict]) -> List[Dict]:
	groups: Dict[int, List[Dict]] = defaultdict(list)
	for r in records:
		groups[r["raw_idx"]].append(r)

	summary: List[Dict] = []
	for raw_idx in sorted(groups.keys()):
		rows = sorted(groups[raw_idx], key=lambda x: x["warp_idx"])
		ratios = [r["ratio"] for r in rows]
		inliers = [r["inliers"] for r in rows]
		totals = [r["total"] for r in rows]
		summary.append(
			{
				"raw_idx": raw_idx,
				"count": len(rows),
				"mean_ratio": mean(ratios),
				"std_ratio": pstdev(ratios) if len(ratios) > 1 else 0.0,
				"min_ratio": min(ratios),
				"max_ratio": max(ratios),
				"weighted_ratio": (sum(inliers) / sum(totals)) if sum(totals) > 0 else 0.0,
				"mean_inliers": mean(inliers),
				"mean_total": mean(totals),
			}
		)
	return summary


def summarize_by_warp(records: List[Dict]) -> List[Dict]:
	warps: Dict[int, List[Dict]] = defaultdict(list)
	for r in records:
		warps[r["warp_idx"]].append(r)

	summary: List[Dict] = []
	for warp_idx in sorted(warps.keys()):
		rows = warps[warp_idx]
		ratios = [r["ratio"] for r in rows]
		inliers = [r["inliers"] for r in rows]
		totals = [r["total"] for r in rows]
		summary.append(
			{
				"warp_idx": warp_idx,
				"count": len(rows),
				"mean_ratio": mean(ratios),
				"std_ratio": pstdev(ratios) if len(ratios) > 1 else 0.0,
				"weighted_ratio": (sum(inliers) / sum(totals)) if sum(totals) > 0 else 0.0,
				"mean_inliers": mean(inliers),
			}
		)
	return summary


def write_csv(path: str, rows: List[Dict], fieldnames: List[str]) -> None:
	with open(path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in rows:
			writer.writerow(row)


def plot_group_mean(summary: List[Dict], out_dir: str) -> None:
	x = [s["raw_idx"] for s in summary]
	y = [s["mean_ratio"] for s in summary]
	yerr = [s["std_ratio"] for s in summary]
	yw = [s["weighted_ratio"] for s in summary]

	plt.figure(figsize=(10, 5))
	plt.bar(x, y, yerr=yerr, capsize=4, alpha=0.75, label="Mean ratio")
	plt.plot(x, yw, marker="o", linewidth=2, label="Weighted ratio")
	plt.ylim(0, 1)
	plt.xticks(x, [f"raw_{i:02d}" for i in x])
	plt.ylabel("Inlier ratio")
	plt.title("Group-level ratio summary")
	plt.grid(axis="y", alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "group_mean_ratio.png"), dpi=160)
	plt.close()


def plot_group_box(records: List[Dict], out_dir: str) -> None:
	groups: Dict[int, List[float]] = defaultdict(list)
	for r in records:
		groups[r["raw_idx"]].append(r["ratio"])

	raw_keys = sorted(groups.keys())
	data = [groups[k] for k in raw_keys]

	plt.figure(figsize=(10, 5))
	plt.boxplot(data, labels=[f"raw_{k:02d}" for k in raw_keys], showmeans=True)
	plt.ylim(0, 1)
	plt.ylabel("Inlier ratio")
	plt.title("Ratio distribution per group")
	plt.grid(axis="y", alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "group_ratio_boxplot.png"), dpi=160)
	plt.close()


def plot_ratio_heatmap(records: List[Dict], out_dir: str) -> None:
	raw_ids = sorted({r["raw_idx"] for r in records})
	warp_ids = sorted({r["warp_idx"] for r in records})

	mat = np.full((len(raw_ids), len(warp_ids)), np.nan)
	rpos = {rid: i for i, rid in enumerate(raw_ids)}
	wpos = {wid: j for j, wid in enumerate(warp_ids)}

	for r in records:
		mat[rpos[r["raw_idx"]], wpos[r["warp_idx"]]] = r["ratio"]

	plt.figure(figsize=(9, 6))
	im = plt.imshow(mat, aspect="auto", cmap="RdYlBu")
	plt.colorbar(im, label="Inlier ratio")
	plt.xticks(range(len(warp_ids)), [f"warp{w}" for w in warp_ids])
	plt.yticks(range(len(raw_ids)), [f"raw_{r:02d}" for r in raw_ids])
	plt.title("Inlier ratio heatmap")

	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			if np.isfinite(mat[i, j]):
				plt.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7)

	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "ratio_heatmap.png"), dpi=180)
	plt.close()


def plot_warp_trend(warp_summary: List[Dict], out_dir: str) -> None:
	x = [w["warp_idx"] for w in warp_summary]
	y = [w["mean_ratio"] for w in warp_summary]
	yerr = [w["std_ratio"] for w in warp_summary]

	plt.figure(figsize=(8, 4.5))
	plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=2, capsize=4)
	plt.ylim(0, 1)
	plt.xticks(x, [f"warp{i}" for i in x])
	plt.ylabel("Mean inlier ratio")
	plt.title("Warp index trend across all groups")
	plt.grid(alpha=0.3)
	plt.tight_layout()
	plt.savefig(os.path.join(out_dir, "warp_trend.png"), dpi=160)
	plt.close()


def print_summary(group_summary: List[Dict], warp_summary: List[Dict], records: List[Dict]) -> None:
	ratios = [r["ratio"] for r in records]
	overall_mean = mean(ratios)
	overall_std = pstdev(ratios) if len(ratios) > 1 else 0.0

	print("\n=== Suggested core metrics ===")
	print("1) Group mean ratio (quality level per raw_0X)")
	print("2) Group std ratio (stability across 6 warps)")
	print("3) Group weighted ratio=sum(inliers)/sum(total) (match-volume-aware quality)")
	print("4) Warp index trend (whether certain warp levels are systematically harder)")

	print("\n=== Overall ===")
	print(f"Total images: {len(records)}")
	print(f"Overall ratio mean/std: {overall_mean:.4f} / {overall_std:.4f}")

	print("\n=== Group ranking by mean ratio ===")
	ranked = sorted(group_summary, key=lambda x: x["mean_ratio"], reverse=True)
	for s in ranked:
		print(
			f"raw_{s['raw_idx']:02d}: mean={s['mean_ratio']:.4f}, std={s['std_ratio']:.4f}, "
			f"weighted={s['weighted_ratio']:.4f}, n={s['count']}"
		)

	print("\n=== Warp trend ===")
	for w in warp_summary:
		print(
			f"warp{w['warp_idx']}: mean={w['mean_ratio']:.4f}, std={w['std_ratio']:.4f}, "
			f"weighted={w['weighted_ratio']:.4f}"
		)


def main() -> None:
	args = parse_args()
	os.makedirs(args.out_dir, exist_ok=True)

	records = load_records(args.csv_path)
	if not records:
		raise ValueError(f"No valid records parsed from {args.csv_path}")

	group_summary = summarize_by_group(records)
	warp_summary = summarize_by_warp(records)

	write_csv(
		os.path.join(args.out_dir, "group_summary.csv"),
		group_summary,
		[
			"raw_idx",
			"count",
			"mean_ratio",
			"std_ratio",
			"min_ratio",
			"max_ratio",
			"weighted_ratio",
			"mean_inliers",
			"mean_total",
		],
	)
	write_csv(
		os.path.join(args.out_dir, "warp_summary.csv"),
		warp_summary,
		["warp_idx", "count", "mean_ratio", "std_ratio", "weighted_ratio", "mean_inliers"],
	)

	plot_group_mean(group_summary, args.out_dir)
	plot_group_box(records, args.out_dir)
	plot_ratio_heatmap(records, args.out_dir)
	plot_warp_trend(warp_summary, args.out_dir)

	print_summary(group_summary, warp_summary, records)
	print(f"\nSaved analysis artifacts to: {args.out_dir}")


if __name__ == "__main__":
	main()
