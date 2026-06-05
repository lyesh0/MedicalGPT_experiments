"""Extract tensorboard logs and generate training curves for SFT/DPO/RM/RLOO/GRPO."""
import argparse
import os
import glob
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tb_scalars(logdir, tag_filter=None):
    """Load all scalar tags from tensorboard logdir into a dict of {tag: [(step, value), ...]}."""
    event_files = glob.glob(os.path.join(logdir, "**", "events.out.*"), recursive=True)
    if not event_files:
        print(f"[WARN] No event files found in {logdir}")
        return {}
    ea = EventAccumulator(os.path.dirname(event_files[0]))
    ea.Reload()
    scalars = {}
    for tag in ea.Tags().get("scalars", []):
        if tag_filter and not tag_filter(tag):
            continue
        events = ea.Scalars(tag)
        scalars[tag] = [(e.step, e.value) for e in events]
    return scalars


def plot_sft(logdir, output):
    scalars = load_tb_scalars(logdir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # train loss
    ax = axes[0]
    for tag in ["train/loss", "loss"]:
        if tag in scalars:
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Train Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # eval loss
    ax = axes[1]
    for tag in ["eval/loss", "eval_loss"]:
        if tag in scalars:
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, "o-", label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Eval Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    # learning rate
    ax = axes[2]
    for tag in ["train/learning_rate", "learning_rate"]:
        if tag in scalars:
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("SFT Training Curves", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_dpo(logdir, output):
    scalars = load_tb_scalars(logdir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    for tag in scalars:
        if "loss" in tag.lower():
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Train Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for tag in scalars:
        if "eval" in tag.lower() and "loss" in tag.lower():
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, "o-", label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Eval Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for tag in ["train/learning_rate", "learning_rate"]:
        if tag in scalars:
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("LR"); ax.set_title("Learning Rate")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("DPO Training Curves", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_rm(logdir, output):
    scalars = load_tb_scalars(logdir)
    n_cols = 4
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    ax = axes[0]
    for tag in scalars:
        if "train" in tag.lower() and "loss" in tag.lower():
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Train Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for tag in scalars:
        if "eval" in tag.lower() and "loss" in tag.lower():
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, "o-", label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss"); ax.set_title("Eval Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for tag in scalars:
        if "accuracy" in tag.lower():
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, "o-", label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Accuracy"); ax.set_title("Pairwise Accuracy")
    ax.legend(); ax.grid(True, alpha=0.3)

    ax = axes[3]
    for tag in scalars:
        if "gap" in tag.lower() or "margin" in tag.lower():
            steps, vals = zip(*scalars[tag])
            ax.plot(steps, vals, "o-", label=tag)
    ax.set_xlabel("Step"); ax.set_ylabel("Gap"); ax.set_title("Reward Gap")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("RM Training Curves", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_rloo(logdir, output):
    scalars = load_tb_scalars(logdir)
    n_cols = 4
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    tags_map = {
        0: ["eval_reward", "reward"],
        1: ["reward_std"],
        2: ["kl", "KL"],
        3: ["mean_length", "length"],
    }
    titles = ["Eval Reward", "Reward Std", "KL Divergence", "Mean Completion Length"]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        for pattern in tags_map[i]:
            for tag in scalars:
                if pattern.lower() in tag.lower():
                    steps, vals = zip(*scalars[tag])
                    ax.plot(steps, vals, "o-", label=tag)
        ax.set_xlabel("Step"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("RLOO Training Curves", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


def plot_grpo(logdir, output):
    scalars = load_tb_scalars(logdir)
    n_cols = 5
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    tags_map = {
        0: ["reward"],
        1: ["reward_std"],
        2: ["kl", "KL"],
        3: ["entropy"],
        4: ["mean_length", "completion_length", "length"],
    }
    titles = ["Reward", "Reward Std", "KL Divergence", "Entropy", "Mean Completion Length"]

    for i, (ax, title) in enumerate(zip(axes, titles)):
        for pattern in tags_map[i]:
            for tag in scalars:
                if pattern.lower() in tag.lower():
                    steps, vals = zip(*scalars[tag])
                    ax.plot(steps, vals, "o-", label=tag)
        ax.set_xlabel("Step"); ax.set_ylabel(title); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("GRPO Training Curves", fontsize=14)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output}")


PLOTTERS = {
    "sft": plot_sft,
    "dpo": plot_dpo,
    "rm": plot_rm,
    "rloo": plot_rloo,
    "grpo": plot_grpo,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", required=True, choices=list(PLOTTERS.keys()))
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    PLOTTERS[args.type](args.logdir, args.output)
