import argparse
import csv

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--csv", type=str, required=True)
parser.add_argument("--output", type=str, default="training_loss.png")
args = parser.parse_args()

train_steps: list[int] = []
train_losses: list[float] = []
val_steps: list[int] = []
val_losses: list[float] = []

with open(args.csv) as f:
    reader = csv.DictReader(f)

    for row in reader:
        step: int = int(row["step"])
        loss: float = float(row["loss"])

        if row["type"] == "train":
            train_steps.append(step)
            train_losses.append(loss)
        elif row["type"] == "val":
            val_steps.append(step)
            val_losses.append(loss)

plt.plot(train_steps, train_losses, label="Training loss")
plt.plot(val_steps, val_losses, label="Validation loss")

plt.xlabel("Training iteration")
plt.ylabel("Cross-entropy loss")
plt.title("TinyStories Training")
plt.legend()
plt.grid(True)

plt.savefig(args.output, dpi=150, bbox_inches="tight")
plt.close()

print(f"Saved plot to {args.output}")

