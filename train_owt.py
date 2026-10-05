from pathlib import Path
import time

from tests.adapters import run_train_bpe


def main() -> None:
    data_path: Path = Path("data/owt_train.txt")

    start_time: float = time.perf_counter()

    vocab, merges = run_train_bpe(
        input_path=data_path,
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
    )

    elapsed: float = time.perf_counter() - start_time

    longest_id: int = max(
            vocab,
            key=lambda token_id: len(vocab[token_id]),
    )

    longest_token: bytes = vocab[longest_id]

    print(f"Training time: {elapsed:.2f} seconds")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(f"Longest token: {longest_token!r}")
    print(f"Longest token length: {len(longest_token)}")


if __name__ == "__main__":
    main()
