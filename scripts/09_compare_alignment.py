import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run_eval(script_path, eval_set, json_out, alignment=None):
    cmd = [
        sys.executable,
        str(script_path),
        "--set",
        eval_set,
        "--json-out",
        str(json_out),
    ]
    if alignment:
        cmd.extend(["--alignment", alignment])
    subprocess.run(cmd, check=True)
    with open(json_out, "r") as f:
        return json.load(f)


def _metric(report, section, key):
    return report.get(section, {}).get(key, None)


def _fmt(v):
    if v is None:
        return "N/A"
    return f"{float(v):.4f}"


def main():
    parser = argparse.ArgumentParser(description="Run baseline vs strict_minonicc alignment comparison.")
    parser.add_argument("--out-dir", type=str, default="data/reports/alignment_compare")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    eval_script = root / "scripts" / "07_evaluate_ranking.py"
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = {}
    for mode_name, alignment in [("baseline", None), ("strict_minonicc", "strict_minonicc")]:
        for eval_set in ("val", "test"):
            out_json = out_dir / f"{mode_name}_{eval_set}.json"
            reports[(mode_name, eval_set)] = _run_eval(
                script_path=eval_script,
                eval_set=eval_set,
                json_out=out_json,
                alignment=alignment,
            )

    rows = []
    metrics = ["CandidatePool_HR@50", "Ranking_HR@50", "Ranking_MRR"]
    for eval_set in ("val", "test"):
        for mode_name in ("baseline", "strict_minonicc"):
            r = reports[(mode_name, eval_set)]
            rows.append(
                {
                    "mode": mode_name,
                    "set": eval_set,
                    "users": r.get("num_users"),
                    "raw_target_in_pool": r.get("target_in_raw_pool"),
                    "force_inserted": r.get("force_inserted_count"),
                    "cap_hr50": _metric(r, "ranking_capability", metrics[0]),
                    "cap_rank_hr50": _metric(r, "ranking_capability", metrics[1]),
                    "cap_mrr": _metric(r, "ranking_capability", metrics[2]),
                    "e2e_hr50": _metric(r, "end_to_end", metrics[0]),
                    "e2e_rank_hr50": _metric(r, "end_to_end", metrics[1]),
                    "e2e_mrr": _metric(r, "end_to_end", metrics[2]),
                }
            )

    md_lines = [
        "# Alignment Comparison Report",
        "",
        "| Mode | Set | Users | RawTargetInPool | ForceInserted | CapPoolHR@50 | CapRankHR@50 | CapMRR | E2EPoolHR@50 | E2ERankHR@50 | E2EMRR |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        md_lines.append(
            f"| {row['mode']} | {row['set']} | {row['users']} | {row['raw_target_in_pool']} | "
            f"{row['force_inserted']} | {_fmt(row['cap_hr50'])} | {_fmt(row['cap_rank_hr50'])} | {_fmt(row['cap_mrr'])} | "
            f"{_fmt(row['e2e_hr50'])} | {_fmt(row['e2e_rank_hr50'])} | {_fmt(row['e2e_mrr'])} |"
        )
    out_md = out_dir / "summary.md"
    out_md.write_text("\n".join(md_lines))
    print(f"Saved comparison report: {out_md}")


if __name__ == "__main__":
    main()

