import subprocess

import modal

app = modal.App("cs336-decoder")

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
    volumes={"/data": volume},
    timeout=7200,
)
def decode(
    prompt: str,
    top_p: float,
    max_num_tokens: int,
    temperature: float,
    run_name: str) -> None:
    import os

    os.chdir("/app")

    command: list[str] = [
        "python",
        "-u",
        "decoder.py",
        "--prompt", prompt,
        "--vocab-path", "/data/owt_train.vocab",
        "--merges-path", "/data/owt_train.merges",
        "--checkpoint-path", "/data/owt_large_lr_1e-3_final.pt",
        "--top-p", str(top_p),
        "--temperature", str(temperature),
        "--max-num-tokens", str(max_num_tokens),
    ]

    print(f"Starting experiment: {run_name}", flush=True)

    try:
        subprocess.run(command, check=True)
    finally:
        volume.commit()

@app.local_entrypoint()
def main(
    prompt: str,
    top_p: float,
    max_num_tokens: int,
    temperature: float,
    run_name: str = "decoder",
    )-> None:
    decode.remote(prompt, top_p, max_num_tokens, temperature, run_name)

