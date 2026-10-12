from pathlib import Path
from bpe.tokenizer import Tokenizer
from tests.common import first_10_docs
import time
from datetime import timedelta
import numpy as np
from multiprocessing import Pool
import os
from bpe.bpe import find_chunk_boundaries

TinyStories_vocab_path = Path("data/TinyStoriesV2-GPT4-train.vocab")
TinyStories_merges_path = Path("data/TinyStoriesV2-GPT4-train.merges")
TinyStories_data_path = Path("data/TinyStoriesV2-GPT4-train.txt")
TinyStories_data_valid_path = Path("data/TinyStoriesV2-GPT4-valid.txt")
TinyStoreis_tokenizer = Tokenizer.from_files(
    str(TinyStories_vocab_path.absolute()),
    str(TinyStories_merges_path.absolute()),
    ["<|endoftext|>"],
)

OWT_vocab_path = Path("data/owt_train.vocab")
OWT_merges_path = Path("data/owt_train.merges")
OWT_data_path = Path("data/owt_train.txt")
OWT_data_valid_path = Path("data/owt_valid.txt")
OWT_tokenizer = Tokenizer.from_files(
    str(OWT_vocab_path.absolute()),
    str(OWT_merges_path.absolute()),
    ["<|endoftext|>"],
)

def sample_10_docs(tokenizer: Tokenizer, data_path: str):
    docs = first_10_docs(data_path)

    start = time.perf_counter()
    tot_tokens = 0
    tot_bytes = 0
    for doc in docs:
        tokens = tokenizer.encode(doc)
        tot_tokens += len(tokens)
        tot_bytes += len(doc.encode("utf-8"))

    time_taken = time.perf_counter() - start
    throughput = tot_bytes / time_taken

    return (tot_bytes / tot_tokens, throughput)

def encode_numpy_worker(
    worker_id:int,
    tokenizer: Tokenizer,
    input_path: str,
    chunks: list[tuple[int,int]],
) -> str:
    print("[worker ", worker_id,  ": start encode numpy file ", input_path)
    output_path = input_path + "-worker" + str(worker_id) + ".npy"
    with open(input_path, "rb") as f:
        # try to break the chunk into 100 segments so that we can
        # use tokenizer.encodeiter to save memory
        print("[worker ", worker_id,  ": find sub segments ", len(chunks))
        
        ids: list[int] = []

        # load segment one by one
        cur:int = 0
        for segment_start, segment_end in chunks:
            print("[worker ", worker_id,  ": start chunk ", cur, "/", len(chunks))
            cur+=1

            f.seek(segment_start)
            chunk: bytes = f.read(segment_end - segment_start)
            local_ids = tokenizer.encode(chunk.decode("utf-8"))
            ids.extend(local_ids)

        array: np.ndarray = np.array(ids, dtype=np.uint16)
        np.save(output_path, array)
        print("[worker ", worker_id,  ": saved total ids ", len(array))
    return output_path

def encode_numpy(
    tokenizer: Tokenizer,
    input_path: str,
    chunks_multiplier: int,
    output_path: str,
):
    num_cores = os.cpu_count()
    desired_num_chunks = num_cores * chunks_multiplier
    boundaries: list[int] = []
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, desired_num_chunks, b"<|endoftext|>")

    chunks: list[tuple[int,int]] = list(zip(boundaries, boundaries[1:]))
    worker_size = (len(chunks) + num_cores - 1) // num_cores
    assert worker_size > 0
    # split into each worker
    worker_segments: list[list[tuple[int,int]]] = [
        chunks[i:i+worker_size]
        for i in range(0, len(chunks), worker_size)
    ]

    print("desired_num_chunks: ", desired_num_chunks)
    print("num_coures: ", num_cores)
    print("chunks_multiplier: ",chunks_multiplier)
    print("actual number of workers: ", len(worker_segments))
    print("actual number of chunks ", len(boundaries))

    args: list[tuple[int, Tokenizer, str, list[tuple[int,int]]]] = [
        (
            i,
            tokenizer,
            input_path,
            worker_segments[i],
        )
        for i in range(len(worker_segments))
    ]
    with Pool(processes = num_cores) as pool:
        all_files = pool.starmap(
            encode_numpy_worker,
            args,
        )
        print("all_files: ", all_files)
        # load each file numpy array and merge them
        arrays: list[np.ndarray] = [
            np.load(path)
            for path in all_files
        ]

        total_tokens: int = sum(len(array) for array in arrays)
        all_ids: np.ndarray = np.empty(total_tokens, dtype=np.uint16)


        print("all files loaded: ", len(all_files), " with num ids=", total_tokens)

        offset: int = 0
        for array in arrays:
            end: int = offset + len(array)
            all_ids[offset:end] = array
            offset = end
        np.save(output_path, all_ids)

        print("starting delete all files...")
        for path in all_files:
            Path(path).unlink(missing_ok=True)

def main() -> None:
    tot_throughput = 0
    ratio, throughput = sample_10_docs(TinyStoreis_tokenizer, TinyStories_data_path)
    print("tiny stories trained tokenizer on tiny story samples gets compression ratio ", ratio, ", throughput=", throughput)
    tot_throughput += throughput

    ratio, throughput = sample_10_docs(OWT_tokenizer, OWT_data_path)
    print("owt trained tokenizer on owt samples gets compression ratio ", ratio, ", throughput=", throughput)
    tot_throughput += throughput

    ratio, throughput = sample_10_docs(TinyStoreis_tokenizer, OWT_data_path)
    print("tiny stories trained tokenizer on owt samples gets compression ratio ", ratio, ", throughput=", throughput)
    tot_throughput += throughput

    avg_throughput = tot_throughput / 3

    projected_bytes = 825 * 10**9
    projected_seconds = projected_bytes / avg_throughput

    print("The estimated time to tokenize 825GB text is ", timedelta(seconds=projected_seconds))

    encode_numpy(
        OWT_tokenizer,
        str(OWT_data_path.absolute()),
        100,
        str(Path("data/owt_train.npy").absolute()),
    )


if __name__ == "__main__":
    main()
