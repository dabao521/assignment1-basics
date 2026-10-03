import os
from typing import BinaryIO
from collections import Counter
import re
import regex
from multiprocessing import Pool

GPT2_PATTERN_STR: str = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT2_PATTERN: regex.Pattern[str] = regex.compile(GPT2_PATTERN_STR)

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def pretokenize_segment(
    segment: bytes,
    local_counter: Counter[tuple[int, ...]],
    ) -> None:
    segment_str: str = segment.decode("utf-8")
    segment_match: regex.Match[str]
    for segment_match in GPT2_PATTERN.finditer(segment_str):
        # print(repr(segment_match.group()))
        token_bytes: bytes = segment_match.group().encode("utf-8")
        # convert to ints
        local_counter[tuple(token_bytes)] += 1

def pretokenize_chunk(
    input_path: str | os.PathLike,
    start: int,
    end: int,
    special_tokens: list[bytes],
) -> Counter[tuple[int, ...]]:
    # shouldn't even start pre tokenizing an empty chunk
    assert end > start, "why pre-tokenize an empty chunk?"

    # build a split pattern from special_tokens
    pattern: re.Pattern[bytes] = re.compile(
            b'|'.join(re.escape(token) for token in special_tokens)
    )
    # 1. read the chunk into str
    local_counter: Counter[tuple[int, ...]] = Counter()
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk: bytes = f.read(end - start)
        # 2. split the chunk bytes into segments based on special_tokens
        # we are using finditer to avoid storing multiple cooy of the bytes
        last_end: int = 0
        match: re.Match[bytes]
        for match in pattern.finditer(chunk):
            # match.start and end() is matching a split token, so [last_end, match.start)
            # is the current valid chunk
            segment: bytes = chunk[last_end:match.start()]
            last_end = match.end()

            # 3.split the segment based on gpt2 pattern and update the token freq map
            # we have to decode back to string since GPT2 pattern works not right
            # in bytes pattern
            if segment:
                pretokenize_segment(segment, local_counter)

        # process the last chunk
        if (last_end < len(chunk)):
            segment: bytes = chunk[last_end:]
            pretokenize_segment(segment, local_counter)

    return local_counter

def pretokenization(
    input_path: str | os.PathLike,
    special_tokens: list[bytes],
    num_processes: int,
) -> Counter[tuple[int, ...]]:
    """
    Main entry to pre-tokenize the file.  It will delegate to pretokenize_chunk for each
    chunk to utilize multi-process cpu architecture

    Args:
        special_tokens: chunk spliter but doesn't have to repsect all of them. If so
        further split witin chunk

    Returns:
        A frequency map for token. The token is represented in tuple(int,...) converted
        from bytes. This is to make the merge process more efficient without copying
        bytes around.
    """
    global_counter: Counter[tuple[int, ...]] = Counter()
    # 1. we use the number of worker proportional to the number of cores so that
    # pretokenization can be parallelized.
    assert len(special_tokens) > 0
    with open(input_path, "rb") as file:
        # 2. get the desired chunks and delegate each worker to do the actual hard work
        boundaries: list[int] = find_chunk_boundaries(
                file, num_processes * 3, special_tokens[0])
        if len(boundaries) == 0:
            return global_counter
        assert len(boundaries) >= 2

        test_args: list[tuple[str | os.PathLike, int, int, list[bytes]]] = [
            (
                input_path,
                boundaries[i],
                boundaries[i+1],
                special_tokens
            )
            for i in range(len(boundaries) - 1)
        ]

        with Pool(processes = num_processes) as pool:
            results: list[Counter[tuple[int, ...]]] = pool.starmap(
                    pretokenize_chunk, test_args)

        # 3. post aggregate local counters to the global counter
        for local_counter in results:
            for token, freq in local_counter.items():
                global_counter[token] += freq

    return global_counter

