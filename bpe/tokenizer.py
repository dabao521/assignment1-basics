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

    @staticmethod
    def remove_node(node:Node):
        node.next_node.prev_node = node.prev_node
        node.prev_node.next_node = node.next_node
        node.prev_node = node
        node.next_node = node

    @staticmethod
    def insert_node(prev_node:Node, node:Node):
        node.next_node = prev_node.next_node
        node.prev_node = prev_node
        node.next_node.prev_node = node
        node.prev_node.next_node = node

    @staticmethod
    def concat(head1: Node | None, head2: Node | None)->Node:
        if head1 is None:
            return head2
        if head2 is None:
            return head1

        tail1 = head1.prev_node
        tail2 = head2.prev_node

        tail1.next_node = head2
        tail1.next_node.prev_node = tail1
        tail2.next_node = head1
        tail2.next_node.prev_node = tail2

        return head1

class Tokenizer:
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None

    # Misc datastructures
    vocab_reverse: dict[bytes, int]
    merge_ranks: dict[tuple[bytes,bytes], int]

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

    def encode(self, text: str) -> list[int]:
        # 1. split text based on special_tokens if any
        merge_head: Node | None = None
        if (self.special_tokens is not None
            and self.special_tokens
           ):
            pattern: regex.Pattern[str] = regex.compile(
                "|".join(regex.escape(t) for t in self.special_tokens)
            )
            # append nodes for all chunks
            last_end: int = 0

            for match in pattern.finditer(text):
                chunk_str: str = text[last_end:match.start()]
                local_head = self.encode_impl(chunk_str)
                # append the special token
                special_token: bytes = match.group().encode("utf-8")
                last_end = match.end()

                # insert the head if it is not empty
                merge_head = Node.concat(merge_head, local_head)

                # insert the special token
                merge_head = Node.append_node(merge_head, Node(special_token))

            # handle the last chunk if any
            if last_end < len(text):
                local_head = self.encode_impl(text[last_end:])
                merge_head = Node.concat(merge_head, local_head)
        else:
            merge_head = self.encode_impl(text)

        # 3. now all pairs merged in the head chain, then iterate the chain and
        # looup encoded integer from the node token bytes.
        encoding: list[int] = []
        cur: Node | None = merge_head

        while (
            cur is not None
            and cur.next_node is not merge_head
            ):
            encoding.append(self.vocab_reverse[cur.token])
            cur = cur.next_node

        if cur is not None:
            encoding.append(self.vocab_reverse[cur.token])
        return encoding

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        token_bytes: bytes = b"".join(
                self.vocab[token_id]
                for token_id in ids
        )
        return token_bytes.decode("utf-8", errors="replace")

    def merge(self, head: Node, min_heap: [tuple(int, Node, Node)])->Node:
        while min_heap:
            rank,node,next_node = heapq.heappop(min_heap)
            # the node can't be at the end or its remembered
            # pair nodes have been removed
            if (node.next_node is node
                or node.next_node is head
                or node.next_node is not next_node):
                continue
            
            key: tuple[bytes, bytes] = (node.token, node.next_node.token)
            assert rank == self.merge_ranks[key]
            new_key: bytes = node.token + node.next_node.token
            new_node: Node = Node(new_key)

            
            # insert first
            Node.insert_node(node.next_node, new_node)

            remove_node1 = node
            remove_node2 = node.next_node
            Node.remove_node(remove_node1)
            Node.remove_node(remove_node2)

            # if we just removed the head, then update the head
            if remove_node1 is head:
                head = new_node

            # find new piars if there is any
            if new_node.next_node is not head:
                self.update_heap(new_node, new_node.next_node, min_heap)

            if new_node is not head:
                self.update_heap(new_node.prev_node, new_node, min_heap)
        return head

    def update_heap(
        self,
        node1: Node,
        node2: Node,
        min_heap: [tuple(int, Node, Node)]):
        key: tuple[bytes,bytes] = (node1.token, node2.token)
        if key in self.merge_ranks:
            heapq.heappush(
                min_heap,
                (self.merge_ranks[key], node1, node2)
            )

    def encode_impl(self, text:str)-> Node:
        """
        encoding the input text into a double linked list node. Merges are applied
        based on the mergs mapping
        """
        merge_head: Node | None = None
        for match in GPT2_PATTERN.finditer(text):
            local_head : Node | Node = None
            min_heap: [tuple(int, Node, Node)] = []

            token: bytes = match.group().encode("utf-8")
            # each byte deserves its own node
            for b in token:
                node: Node = Node(bytes([b]))
                local_head = Node.append_node(local_head, node)

                # validate if there is a pair so we need to push onto
                # the min_heap
                if node.prev_node is not node:
                    key: tuple[bytes, bytes] = (node.prev_node.token, node.token)
                    if key in self.merge_ranks:
                        heapq.heappush(
                            min_heap,
                            (self.merge_ranks[key], node.prev_node, node)
                        )
            assert local_head is not None, "gpt2 matched an empty string"
            local_head = self.merge(local_head, min_heap)
            merge_head = Node.concat(merge_head, local_head)
        return merge_head

