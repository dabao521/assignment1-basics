import subprocess

import modal

app = modal.App("cs336-learning-rate-sweep")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync("./")
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            ".venv/**",
            ".git/**",
            "data/**",
            "*.npy",
            "*.pt",
            "*.csv",
            "*.png",
            "__pycache__/**",
        ],
    )
)

volume = modal.Volume.from_name("cs336-data")

@app.function(
    image=image,
    gpu="B200",
    volumes={"/data": volume},
    timeout=7200,
)
def train(max_lr: float, batch_size: int, num_iterations: int, run_name: str) -> None:
    import os

    os.chdir("/app")

    command: list[str] = [
        "python",
        "-u",
        "train_loop.py",
        "--train-data", "/data/owt_train.npy",
        "--val-data", "/data/owt_valid.npy",
        "--vocab-size", "32000",
        "--context-length", "512",
        "--d-model", "640",
        "--num-layers", "12",
        "--num-heads", "10",
        "--d-ff", "1728",
        "--theta", "10000",
        "--batch-size", str(batch_size),
        "--num-iterations", str(num_iterations),
        "--device", "cuda",
        "--beta1", "0.9",
        "--beta2", "0.999",
        "--weight-decay", "0.01",
        "--max-learning-rate", str(max_lr),
        "--min-learning-rate", "1e-4",
        "--warmup-iters", "200",
        "--cosine-cycle-iters", "22000",
        "--max-l2-norm", "1.0",
        "--log-interval", "100",
        "--eval-interval", "1000",
        "--eval-batches", "10",
        "--log-file", f"/data/{run_name}.csv",
        "--checkpoint-file", f"/data/{run_name}.pt",
        "--checkpoint-interval", "100001",
    ]

    print(f"Starting experiment: {run_name}", flush=True)

    try:
        subprocess.run(command, check=True)
    finally:
        volume.commit()

@app.local_entrypoint()
def main(
    max_lr: float = 0.001,
    batch_size: int = 64,
    num_iterations: int = 20000,
    run_name: str = "lr_1e-3",
    )-> None:
    train.remote(max_lr, batch_size, num_iterations, run_name)

