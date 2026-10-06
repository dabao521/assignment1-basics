from __future__ import annotations
import heapq
import regex
from collections.abc import Iterable
from collections.abc import Iterator

from bpe.bpe import GPT2_PATTERN

class Node:
    token: bytes
    prev_node: Node | None
    next_node: Node | None

    def __init__(
        self,
        token: bytes,
    ):
        self.token = token
        self.prev_node = self
        self.next_node = self

    def __lt__(self, other: Node)->bool:
        return id(self) < id(other)

class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None

    # Misc datastructures
    vocab_reverse: dict[bytes, int]
    merge_ranks: dict[tuple[bytes,bytes], int]

    min_heap: [tuple(int, Node, Node)]
    head: Node | None

    byte_encoder: dict[int, str]
    byte_decoder: dict[str, int]


    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        if special_tokens is not None:
            self.special_tokens = sorted(special_tokens, key=len, reverse=True)
        else:
            self.special_tokens = None

        self.vocab_reverse = {
            token: index
            for index, token in self.vocab.items()
        }
        self.merge_ranks = {
            pair: rank
            for rank, pair in enumerate(self.merges)
        }
        self.min_heap = []
        self.head = None

        self.byte_encoder = Tokenizer.bytes_to_unicode()
        self.byte_decoder = {
            v: k for k, v in self.byte_encoder.items()
        }

    def encode(self, text: str) -> list[int]:
        # 1. split text based on special_tokens if any
        # and intialized the min_heap and the linked list
        self.reset()

        if (self.special_tokens is not None
            and self.special_tokens
           ):
            pattern: regex.Pattern[str] = regex.compile(
                "|".join(regex.escape(t) for t in self.special_tokens)
            )
            # append nodes for all chunks
            last_end: int = 0
            local_head: Node | None = None
            for match in pattern.finditer(text):
                chunk_str: str = text[last_end:match.start()]
                self.append_nodes(chunk_str)
                # append the special token
                special_token: bytes = match.group().encode("utf-8")
                last_end = match.end()

                # insert the head if it is not empty
                if self.head is not None:
                    if local_head is None:
                        local_head = self.head
                    else:
                        Tokenizer.concat(local_head, self.head)

                # insert the special token
                local_head = Tokenizer.append_node(local_head, Node(special_token))

            # handle the last chunk if any
            if last_end < len(text):
                self.append_nodes(text[last_end:])
                if self.head is not None:
                    if local_head is None:
                        local_head = self.head
                    else:
                        Tokenizer.concat(local_head, self.head)
            self.head = local_head
        else:
            self.append_nodes(text)

        # 3. now all pairs merged in the head chain, then iterate the chain and
        # looup encoded integer from the node token bytes.
        encoding: list[int] = []
        cur: Node | None = self.head

        while (
            cur is not None
            and cur.next_node is not self.head
            ):
            encoding.append(self.vocab_reverse[cur.token])
            cur = cur.next_node

        if cur is not None:
            encoding.append(self.vocab_reverse[cur.token])
        return encoding

    @staticmethod
    def concat(head1: Node, head2: Node):
        assert head1 is not None
        assert head2 is not None

        tail1 = head1.prev_node
        tail2 = head2.prev_node

        tail1.next_node = head2
        tail1.next_node.prev_node = tail1
        tail2.next_node = head1
        tail2.next_node.prev_node = tail2

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def reset(self):
        self.min_heap.clear()
        self.head = None

    def decode(self, ids: list[int]) -> str:
        text: str = "".join(
            "".join(self.byte_encoder[b] for b in self.vocab[token_id])
            for token_id in ids
        )
        token_bytes: bytes = bytes(
            self.byte_decoder[c]
            for c in text
        )
        return token_bytes.decode("utf-8", errors="replace")

    def merge(self):
        while self.min_heap:
            rank,node,next_node = heapq.heappop(self.min_heap)
            if (node.next_node is node
                or node.next_node is self.head
                or node.next_node is not next_node):
                # invalid node
                continue
            
            key: tuple[bytes, bytes] = (node.token, node.next_node.token)
            assert rank == self.merge_ranks[key]
            new_key: bytes = node.token + node.next_node.token
            new_node: Node = Node(new_key)

            
            # insert first
            self.insert_node(node.next_node, new_node)

            remove_node1 = node
            remove_node2 = node.next_node
            self.remove_node(remove_node1)
            self.remove_node(remove_node2)

            # if we just removed the head, then update the head
            if remove_node1 is self.head:
                self.head = new_node

            # find new piars if there is any
            if new_node.next_node is not self.head:
                self.update_heap(new_node, new_node.next_node)

            if new_node is not self.head:
                self.update_heap(new_node.prev_node, new_node)

    def update_heap(self, node1: Node, node2: Node):
        key: tuple[bytes,bytes] = (node1.token, node2.token)
        if key in self.merge_ranks:
            heapq.heappush(
                self.min_heap,
                (self.merge_ranks[key], node1, node2)
            )


    def append_nodes(self, text:str):
        local_head: Node | None = None
        for match in GPT2_PATTERN.finditer(text):
            self.reset()
            token: bytes = match.group().encode("utf-8")
            # each byte deserves its own node
            for b in token:
                node: Node = Node(bytes([b]))
                self.head = Tokenizer.append_node(self.head, node)

                # validate if there is a pair so we need to push onto
                # the min_heap
                if node.prev_node is not node:
                    key: tuple[bytes, bytes] = (node.prev_node.token, node.token)
                    if key in self.merge_ranks:
                        heapq.heappush(
                            self.min_heap,
                            (self.merge_ranks[key], node.prev_node, node)
                        )
            self.merge()
            if self.head is not None:
                if local_head is None:
                    local_head = self.head
                else:
                    Tokenizer.concat(local_head, self.head)
        self.head = local_head

    @staticmethod
    def append_node(head: Node | None, node: Node) -> Node:
        if head is None:
           return node 
        else:
            node.next_node = head
            node.prev_node = head.prev_node
            node.next_node.prev_node = node
            node.prev_node.next_node = node
            return head

    def remove_node(self, node:Node):
        node.next_node.prev_node = node.prev_node
        node.prev_node.next_node = node.next_node
        node.prev_node = node
        node.next_node = node

    def insert_node(self, prev_node:Node, node:Node):
        node.next_node = prev_node.next_node
        node.prev_node = prev_node
        node.next_node.prev_node = node
        node.prev_node.next_node = node

    @staticmethod
    def bytes_to_unicode() -> dict[int, str]:
        bs: list[int] = (
            list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1))
        )

        cs: list[int] = bs.copy()
        n: int = 0

        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1

        return dict(zip(bs, (chr(c) for c in cs)))

