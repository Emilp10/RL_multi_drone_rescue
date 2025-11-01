from __future__ import annotations
import argparse
import itertools
import json
import os
import os.path as osp
import shutil
import subprocess
import sys
import time
import csv


def run(cmd: list[str], cwd: str | None = None) -> int:
    print("$", " ".join(cmd))
    p = subprocess.run(cmd, cwd=cwd)
    return p.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", type=int, default=1, choices=[1, 3])
    parser.add_argument("--trials", type=int, default=4, help="Number of trial combinations to run")
    parser.add_argument("--episodes", type=int, default=80, help="Episodes per trial")
    parser.add_argument("--min-episodes", type=int, default=30, help="Min episodes before training starts")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes for evaluation per trial")
    parser.add_argument("--grid", action="store_true", help="Use a small grid instead of random sampling")
    args = parser.parse_args()

    ws = osp.dirname(__file__)
    proj = osp.abspath(osp.join(ws, ".."))

    # Search space (small)
    lr_vals = [1e-4, 3e-4]
    seq_vals = [16, 32]
    nstep_vals = [3, 5]
    decay_vals = [50000, 100000]

    combos = list(itertools.product(lr_vals, seq_vals, nstep_vals, decay_vals))
    if not args.grid:
        # downsample for random trials
        import random
        random.shuffle(combos)
        combos = combos[: args.trials]
    else:
        combos = combos[: args.trials]

    os.makedirs(osp.join(proj, "logs"), exist_ok=True)
    tuning_csv = osp.join(proj, "logs", f"tuning_agents_{args.agents}.csv")
    with open(tuning_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "lr", "seq_len", "n_step", "eps_decay", "episodes", "success", "avg_return", "avg_rescued", "ckpt_dir"])

    best = None
    for i, (lr, seq_len, n_step, decay) in enumerate(combos, start=1):
        suffix = f"tune_{args.agents}_t{i}"
        # Training runs with cwd=ws; checkpoints saved under ws/checkpoints/...
        run_dir = osp.join(ws, "checkpoints", f"drqn_agents_{args.agents}_{suffix}")
        ckpt_final = osp.join(run_dir, "final")
        # Train
        code = run([
            sys.executable,
            "train.py",
            "--agents", str(args.agents),
            "--episodes", str(args.episodes),
            "--min-episodes", str(args.min_episodes),
            "--save-every", str(max(10, args.episodes // 2)),
            "--lr", str(lr),
            "--seq-len", str(seq_len),
            "--n-step", str(n_step),
            "--epsilon-decay-steps", str(decay),
            "--save-suffix", suffix,
        ], cwd=ws)
        if code != 0:
            print(f"Trial {i} training failed; skipping")
            continue

        # Evaluate against the trial ckpt dir
        import evaluate
        # Evaluate the trial's final checkpoint directory directly
        res = evaluate.evaluate_agents(args.agents, episodes=args.eval_episodes, ckpt_dir=ckpt_final)
        summary = res.get("summary", {})
        success = summary.get("success_rate", 0.0)
        avg_ret = summary.get("avg_return", 0.0)
        avg_res = summary.get("avg_rescued", 0.0)

        with open(tuning_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([i, lr, seq_len, n_step, decay, args.episodes, success, avg_ret, avg_res, ckpt_final])

        score = (success, avg_ret)
        if (best is None) or (score > best[0]):
            best = (score, {
                "lr": lr,
                "seq_len": seq_len,
                "n_step": n_step,
                "eps_decay": decay,
                "ckpt_dir": ckpt_final,
                "summary": summary,
            })

    if best is not None:
        print("Best:", json.dumps(best[1], indent=2))
        # Optionally copy best to canonical final for the UI
        best_dir = best[1]["ckpt_dir"]  # this is the "final" directory for the best trial
        final_dir = osp.join(proj, "checkpoints", f"drqn_agents_{args.agents}", "final")
        os.makedirs(osp.dirname(final_dir), exist_ok=True)
        if osp.isdir(final_dir):
            shutil.rmtree(final_dir)
        shutil.copytree(best_dir, final_dir)
        print("Copied best checkpoint to:", final_dir)
    else:
        print("No successful trials recorded.")


if __name__ == "__main__":
    main()
