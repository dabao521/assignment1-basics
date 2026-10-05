from pathlib import Path
import time

from tests.adapters import run_train_bpe


def main() -> None:
    data_path: Path = Path("data/TinyStoriesV2-GPT4-train.txt")

    start_time: float = time.perf_counter()

    vocab, merges = run_train_bpe(
        input_path=data_path,
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed: float = time.perf_counter() - start_time

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")


if __name__ == "__main__":
    main()
