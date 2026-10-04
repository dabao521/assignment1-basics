import json
import time

from .adapters import run_train_bpe
from .common import FIXTURES_PATH, gpt2_bytes_to_unicode
from bpe.bpe import pretokenization
from collections import Counter
import pytest


def test_train_bpe_speed():
    """
    Ensure that BPE training is relatively efficient by measuring training
    time on this small dataset and throwing an error if it takes more than 1.5 seconds.
    This is a pretty generous upper-bound, it takes 0.38 seconds with the
    reference implementation on my laptop. In contrast, the toy implementation
    takes around 3 seconds.
    """
    input_path = FIXTURES_PATH / "corpus.en"
    start_time = time.time()
    _, _ = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )
    end_time = time.time()
    assert end_time - start_time < 1.5


def test_train_bpe():
    input_path = FIXTURES_PATH / "corpus.en"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

    # Path to the reference tokenizer vocab and merges
    reference_vocab_path = FIXTURES_PATH / "train-bpe-reference-vocab.json"
    reference_merges_path = FIXTURES_PATH / "train-bpe-reference-merges.txt"

    # Compare the learned merges to the expected output merges
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}
    with open(reference_merges_path, encoding="utf-8") as f:
        gpt2_reference_merges = [tuple(line.rstrip().split(" ")) for line in f]
        reference_merges = [
            (
                bytes([gpt2_byte_decoder[token] for token in merge_token_1]),
                bytes([gpt2_byte_decoder[token] for token in merge_token_2]),
            )
            for merge_token_1, merge_token_2 in gpt2_reference_merges
        ]
    assert merges == reference_merges

    # Compare the vocab to the expected output vocab
    with open(reference_vocab_path, encoding="utf-8") as f:
        gpt2_reference_vocab = json.load(f)
        reference_vocab = {
            gpt2_vocab_index: bytes([gpt2_byte_decoder[token] for token in gpt2_vocab_item])
            for gpt2_vocab_item, gpt2_vocab_index in gpt2_reference_vocab.items()
        }
    # Rather than checking that the vocabs exactly match (since they could
    # have been constructed differently), we'll make sure that the vocab keys and values match
    assert set(vocab.keys()) == set(reference_vocab.keys())
    assert set(vocab.values()) == set(reference_vocab.values())


def test_train_bpe_special_tokens(snapshot):
    """
    Ensure that the special tokens are added to the vocabulary and not
    merged with other tokens.
    """
    input_path = FIXTURES_PATH / "tinystories_sample_5M.txt"
    vocab, merges = run_train_bpe(
        input_path=input_path,
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )

    # Check that the special token is not in the vocab
    vocabs_without_specials = [word for word in vocab.values() if word != b"<|endoftext|>"]
    for word_bytes in vocabs_without_specials:
        assert b"<|" not in word_bytes

    snapshot.assert_match(
        {
            "vocab_keys": set(vocab.keys()),
            "vocab_values": set(vocab.values()),
            "merges": merges,
        },
    )

def test_pretokenization_basic(tmp_path):
    input_path = tmp_path / "test.txt"

    input_path.write_text(
        "hello world hello",
        encoding = "utf-8")

    actual = pretokenization(
            input_path,
            [b"<|endoftext|>"],
            num_processes=1,
    )

    expected = Counter({
        tuple(b"hello"): 1,
        tuple(b" world"): 1,
        tuple(b" hello"): 1,
    })

    assert actual == expected

@pytest.mark.parametrize("num_processes", [1, 2, 4, 8])
def test_pretokenization_multiple_processes(tmp_path, num_processes):
    input_path = tmp_path / "test.txt"

    input_path.write_text(
            "hello world hello "
            "<|endoftext|> "
            "foo bar foo "
            "<|endoftext|> "
            "hello world",
            encoding="utf-8",
            )

    actual = pretokenization(
            input_path,
            [b"<|endoftext|>"],
            num_processes=num_processes,
            )

    expected = Counter({
        tuple(b"hello"): 1,
        tuple(b" hello"): 2,
        tuple(b" "): 2,
        tuple(b" world"): 2,
        tuple(b" foo"): 2,
        tuple(b" bar"): 1,
        })

    assert actual == expected

def test_pretokenization_multiple_special_tokens(tmp_path):
    input_path = tmp_path / "test.txt"

    input_path.write_text(
        "hello<|endoftext|>world<|endoftext|>hello",
        encoding = "utf-8")

    actual = pretokenization(
            input_path,
            [b"<|endoftext|>"],
            num_processes=1,
    )

    expected = Counter({
        tuple(b"hello"): 2,
        tuple(b"world"): 1,
    })

    assert actual == expected

def test_pretokenization_unicode(tmp_path):
    input_path = tmp_path / "test.txt"

    input_path.write_text(
        "hello 你好 世界 café",
        encoding = "utf-8")

    actual = pretokenization(
            input_path,
            [b"<|endoftext|>"],
            num_processes=1,
    )

    expected = Counter({
        tuple(b"hello"): 1,
        tuple(" 你好".encode()): 1,
        tuple(" 世界".encode()): 1,
        tuple(" café".encode()): 1,
    })

    assert actual == expected
