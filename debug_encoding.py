from pathlib import Path

import numpy as np

from bpe.tokenizer import Tokenizer
# from tokenizer_experiments import TinyStoreis_tokenizer
from tokenizer_experiments import OWT_tokenizer

def debug_decode(
    tokenizer: Tokenizer,
    input_path: str,
    token_path: str,
    output_path: str,
) -> None:
    # Load token IDs
    ids: np.ndarray = np.load(token_path)

    print("Number of tokens:", len(ids))
    print("dtype:", ids.dtype)

    # Convert numpy integers to Python ints
    token_ids: list[int] = ids.tolist()

    # Decode
    decoded: str = tokenizer.decode(token_ids)

    # Write reconstructed text
    Path(output_path).unlink(missing_ok=True)
    Path(output_path).write_text(
        decoded,
        encoding="utf-8",
    )

    # Compare bytes rather than Python strings.
    # This makes the comparison exact.
    original_bytes: bytes = Path(input_path).read_bytes()
    decoded_bytes: bytes = decoded.encode("utf-8")

    print("Original bytes:", len(original_bytes))
    print("Decoded bytes:", len(decoded_bytes))

    if original_bytes == decoded_bytes:
        print("SUCCESS: decoded output is identical to original.")
        return
    
    print("ERROR: decoded output differs from original.")

    common_len: int = min(len(original_bytes), len(decoded_bytes))

    first_diff: int | None = None
            
    for i in range(common_len):
        if original_bytes[i] != decoded_bytes[i]:
            first_diff = i
            break

    if first_diff is None:
        print("One file is a prefix of the other.")
        print("Length difference:", len(decoded_bytes) - len(original_bytes))
        return
    
    print("First byte difference:", first_diff)

    start: int = max(0, first_diff - 100)
    end: int = min(common_len, first_diff + 100)

    print()
    print("Original:")
    print(repr(original_bytes[start:end]))

    print()
    print("Decoded:")
    print(repr(decoded_bytes[start:end]))

    print()
    print(
        "Original byte:",
        original_bytes[first_diff],
        chr(original_bytes[first_diff])
        if original_bytes[first_diff] < 128
        else "",
    )

    print(
        "Decoded byte:",
        decoded_bytes[first_diff],
        chr(decoded_bytes[first_diff])
        if decoded_bytes[first_diff] < 128
        else "",
    )

if __name__ == "__main__":
    tokenizer = OWT_tokenizer

    debug_decode(
        tokenizer,
        "data/owt_valid.txt",
        "data/owt_valid.npy",
        "data/debug_decoded.txt",
    )

