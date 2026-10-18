import argparse
import csv

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--output", type=str, default="training_loss.png")
parser.add_argument(
    "--x-axis",
    choices=["step", "elapsed_seconds"],
    default="step",
)
args = parser.parse_args()

train_x: list[int] = []
train_losses: list[float] = []
val_x: list[int] = []
val_losses: list[float] = []

with open(args.csv) as f:
    reader = csv.DictReader(f)
    rows: list[dict[str, str]] = list(reader)

if not rows:
    raise ValueError("CSV contains no data")


config: dict[str, str] = rows[0]

hyperparam_keys: list[tuple[str, str]] = [
    ("num_layers", "Layers"),
    ("d_model", "d_model"),
    ("d_ff", "d_ff"),
    ("num_heads", "Heads"),
    ("context_length", "Context"),
    ("batch_size", "Batch"),
    ("max_learning_rate", "Max LR"),
    ("min_learning_rate", "Min LR"),
]

hyperparams: list[str] = [
    f"{label}={config[key]}"
    for key, label in hyperparam_keys
    if config.get(key)
]

hyperparam_label: str = " | ".join(hyperparams)

for row in rows:
    x: float = float(row[args.x_axis])
    loss: float = float(row["loss"])

    if row["type"] == "train":
        train_x.append(x)
        train_losses.append(loss)
    elif row["type"] == "val":
        val_x.append(x)
        val_losses.append(loss)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_x, train_losses, label="Training loss")
ax.plot(val_x, val_losses, label="Validation loss")

if args.x_axis == "step":
    ax.set_xlabel("Gradient steps")
else:
    ax.set_xlabel("Wall-clock time (seconds)")

ax.set_ylabel("Cross-entropy loss")

title: str = "TinyStories Training"
if hyperparam_label:
    title += f"\n{hyperparam_label}"

    ax.set_title(title, fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot to {args.output}")

