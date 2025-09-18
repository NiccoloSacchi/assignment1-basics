"""BPE tokenizer. Supports training, decoding, and encoding."""

from typing import Iterable, Iterator

import regex as re
import torch

from cs336_basics import utils


get_tqdm = utils.get_tqdm


class BPETokenizer:
    """Tokenizer that uses byte-pair encoding."""

    def __init__(
        self,
        vocab: dict[int, bytes] | None = None,
        merges: list[tuple[bytes, bytes]] | None = None,
        special_tokens: list[str] | None = None,
    ):
        # Used for mapping token to bytes (decoding) and vice-versa (encoding).
        self.vocab = vocab
        # Used during encoding to know which byte strings should be merged.
        self.merges = merges
        # List of strings that should always be kept as a single token. Used during
        # the pre-tokenization step in the encoding.
        self.special_tokens = special_tokens if special_tokens else []
        # Regex used for pre-tokenization.
        self.pat = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

    def train(
        self,
        text: str,
        vocab_size: int,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        verbose: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train a BPE tokenizer.

        Args:
            text (str): Text containing all the documents used to train the tokenizer.
                Might contain special tokens.
            vocab_size (int): The size of the vocabulary at the end of the training.
            special_tokens (tuple[str, ...] | None): A list of strings that should not
                be split nor merged into tokens. Each string in special_token will be
                added to the vocabulary and get its token.
            verbose (bool): Whether to print debug information.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """
        assert vocab_size > 256 + len(special_tokens), "Vocab size too small"

        if verbose:
            print(f"Training on a text of length {len(text)}...")

        chunk_counts = self._pre_tokenize(
            text=text, special_tokens=special_tokens, verbose=verbose
        )
        vocab, merges = self._compute_merges(chunk_counts, vocab_size, special_tokens, verbose)
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens
        return vocab, merges
    
    def train_iterable(
        self,
        texts: Iterable[str],
        vocab_size: int,
        special_tokens: tuple[str, ...] | list[str] = ("<|endoftext|>",),
        verbose: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Train a BPE tokenizer on a iterable to save memory.

        Args:
            text (str): Text containing all the documents used to train the tokenizer.
                Might contain special tokens.
            vocab_size (int): The size of the vocabulary at the end of the training.
            special_tokens (tuple[str, ...] | None): A list of strings that should not
                be split nor merged into tokens. Each string in special_token will be
                added to the vocabulary and get its token.
            verbose (bool): Whether to print debug information.

        Returns:
            A BPE tokenizer that uses the provided vocab, merges, and special tokens.
        """
        assert vocab_size > 256 + len(special_tokens), "Vocab size too small"

        if verbose:
            print(f"Training on a texts iterable...")

        chunk_counts = self._pre_tokenize(
            texts=texts, special_tokens=special_tokens, verbose=verbose
        )
        vocab, merges = self._compute_merges(chunk_counts, vocab_size, special_tokens, verbose)
        self.merges = merges
        self.vocab = vocab
        self.special_tokens = special_tokens
        return vocab, merges

    def _compute_merges(
        self,
        chunk_counts: dict[tuple[bytes, ...], int] | None = None,
        vocab_size: int | None = None,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        verbose: bool = False,
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        """Compute the most common merges to be performed during BPE training.

        Args:
            chunk_counts (dict[typle[bytes], int]): contains the result of the
                pre-tokenization. Only one of text or chunk_counts should be provided.
            vocab_size (int): The size of the vocabulary at the end of the training.
            special_tokens (tuple[str, ...] | None): A list of strings that should not
                be split nor merged into tokens. Each string in special_token will be
                added to the vocabulary and get its token.
            verbose (bool): Whether to print debug information.

        Returns:
            vocab (dict[int, bytes]): The vocabulary mapping token ids to byte strings.
            merges (list[tuple[bytes, bytes]]): The list of merges to be performed
        """
        # Convert chunk_count to list so that we can index with just an integer.
        chunk_counts = list(
            [[chunk, count] for chunk, count in chunk_counts.items()]
        )
        # Compute the pair_count only once and the update them at every merge
        # instead than recomputing them.
        pair_count = {}
        pair_to_chunk_ids = {}
        for i, (chunk, count) in get_tqdm(
            enumerate(chunk_counts), condition=verbose, desc="Computing pair counts"
        ):
            for j in range(len(chunk) - 1):
                pair = chunk[j : j + 2]
                pair_count[pair] = pair_count.get(pair, 0) + count
                if pair not in pair_to_chunk_ids:
                    pair_to_chunk_ids[pair] = []
                pair_to_chunk_ids[pair].append(i)

        merges = []  # list[tuple(bytes)]
        num_merges = vocab_size - (256 + len(special_tokens))
        for _ in get_tqdm(range(num_merges), condition=verbose, desc="Merging"):
            # Find the lexicographically greater (for determinism) most common pair of
            # byte strings and add it to the merges.
            max_pair = None
            max_count = 0
            for pair, count in pair_count.items():
                if count > max_count or (count == max_count and pair > max_pair):
                    max_pair = pair
                    max_count = count
            if not max_pair:
                if verbose:
                    print(
                        "There is nothing else to merge, stopping after {} merges."
                        .format(len(merges))
                    )
                break
            merges.append(max_pair)

            def _add_pair(pair, count, chunk_id, pair_count, pair_to_chunk_ids):
                pair_count[pair] = pair_count.get(pair, 0) + count
                if pair not in pair_to_chunk_ids:
                    pair_to_chunk_ids[pair] = []
                pair_to_chunk_ids[pair].append(chunk_id)

            # Merge the tokens also in chunk_counts.
            max_pair_merged = max_pair[0] + max_pair[1]
            for i in pair_to_chunk_ids[max_pair]:
                chunk, count = chunk_counts[i]
                j = 0
                while j < len(chunk) - 1:
                    pair = chunk[j : j + 2]
                    if pair != max_pair:
                        j += 1
                        continue
                    # Update the affected pairs on the left and on the right of the merge.
                    left_pair = chunk[j - 1 : j + 1]
                    if len(left_pair) == 2:
                        assert pair_count[left_pair] > 0, "pair_count[left_pair] <= 0"
                        pair_count[left_pair] -= count
                        # We should also drop this occurrence from pair_to_chunk_ids but it
                        # does not introduce a bug so whatever.
                        if pair_count[left_pair] == 0:
                            del pair_count[left_pair]
                            del pair_to_chunk_ids[left_pair]

                    right_pair = chunk[j + 1 : j + 3]
                    if len(right_pair) == 2:
                        assert pair_count[right_pair] > 0, "pair_count[right_pair] <= 0"
                        pair_count[right_pair] -= count
                        # We should also drop this occurrence from pair_to_chunk_ids but it
                        # does not introduce a bug so whatever.
                        if pair_count[right_pair] == 0:
                            del pair_count[right_pair]
                            del pair_to_chunk_ids[right_pair]

                    # Create the merged chunk and register the new pairs.
                    _add_pair(max_pair, count, i, pair_count, pair_to_chunk_ids)
                    chunk = chunk[:j] + (max_pair_merged,) + chunk[j + 2 :]
                    new_left_pair = chunk[j - 1 : j + 1]
                    if len(new_left_pair) == 2:
                        _add_pair(new_left_pair, count, i, pair_count, pair_to_chunk_ids)
                    new_right_pair = chunk[j : j + 2]
                    if len(new_right_pair) == 2:
                        _add_pair(new_right_pair, count, i, pair_count, pair_to_chunk_ids)
                    j += 1
                chunk_counts[i][0] = chunk
            # Delete the count and positions of the pair we merged.
            del pair_count[max_pair]
            del pair_to_chunk_ids[max_pair]

        # Build the vocab for the merges.
        vocab = {i: bytes([i]) for i in range(256)}  # dict[int, bytes]
        for m in merges:
            vocab[len(vocab)] = m[0] + m[1]
        for t in special_tokens:
            vocab[len(vocab)] = t.encode("utf-8")
        return vocab, merges

    def encode_iterable(
        self, texts: Iterable[str], verbose: bool = False
    ) -> Iterator[int]:
        """Lazily encodes an texts of text into an iterable of token ids."""
        assert self.vocab is not None, "missing vocab"
        assert self.merges is not None, "missing merges"
        assert self.special_tokens is not None, "special_tokens is None"
        for text in get_tqdm(texts, condition=verbose, desc="Encoding"):
            for i in self.encode(text):
                yield i

    def encode(self, text: str, verbose: bool = False) -> list[int]:
        """Encodes a text into a list of token ids."""
        assert self.vocab is not None, "missing vocab"
        assert self.merges is not None, "missing merges"
        assert self.special_tokens is not None, "special_tokens is None"

        # Convert from dict[int, bytes] to dict[bytes, int].
        vocab = {b: i for i, b in self.vocab.items()}

        texts = [text]
        if self.special_tokens:
            # Sort to support special tokens that are substrings of other special
            # tokens. For example, if we have special tokens <|endoftext|> and
            # <|endoftext|>123, we want to match the longer one first.
            special_tokens_split_pat = "|".join(
                re.escape(st) for st in sorted(self.special_tokens, key=len, reverse=True)
            )
            # Use capturing parentheses to include the special tokens in the split
            # result.
            texts = re.split(f"({special_tokens_split_pat})", text)

        tokens = []
        special_tokens_set = set(self.special_tokens)

        # For optimization, we create a dict of merges to easily find which merge
        # has priority.
        merge_to_priority = {m: i for i, m in enumerate(self.merges)}
        for text in get_tqdm(texts, condition=verbose, desc="Encoding"):
            if text in special_tokens_set:
                st_bytes = text.encode("utf-8")
                assert st_bytes in vocab, f"special token {text} not in {vocab}"
                tokens.append(vocab[st_bytes])
                continue

            for chunk in re.finditer(self.pat, text):
                chunk_bytes = [bytes([b]) for b in chunk.group().encode("utf-8")]
                while len(chunk_bytes) > 1:
                    # Get all the pairs of bytes in the chunk.
                    pairs = set()
                    for j in range(len(chunk_bytes) - 1):
                        pairs.add(tuple(chunk_bytes[j : j + 2]))

                    # Find the pair with the highest priority.
                    merge_pair = min(
                        pairs, key=lambda p: merge_to_priority.get(p, float("inf"))
                    )
                    if merge_pair not in merge_to_priority:
                        # Nothing else can be merged.
                        break

                    # Merge the pair in the chunk.
                    i = 0
                    new_chunk_bytes = []
                    while i < len(chunk_bytes):
                        if (
                            len(chunk_bytes[i : i + 2]) == 2
                            and chunk_bytes[i] == merge_pair[0]
                            and chunk_bytes[i + 1] == merge_pair[1]
                        ):
                            new_chunk_bytes.append(merge_pair[0] + merge_pair[1])
                            i += 2
                        else:
                            new_chunk_bytes.append(chunk_bytes[i])
                            i += 1
                    chunk_bytes = new_chunk_bytes

                # Finished merging, now convert the bytes to tokens.
                for chunk_byte in chunk_bytes:
                    assert chunk_byte in vocab, f"chunk_byte {chunk_byte} not in {vocab}"
                    tokens.append(vocab[chunk_byte])
        return tokens

    def decode(self, tokens: list[int], verbose: bool = False) -> str:
        """Decodes a list of token ids into a text."""
        assert self.vocab is not None, "missing vocab"
        assert self.merges is not None, "missing merges"
        assert self.special_tokens is not None, "missing special tokens"

        chunk_bytes = []
        for token in get_tqdm(tokens, condition=verbose, desc="Decoding"):
            assert token in self.vocab, f"token {token} not in {self.vocab}"
            chunk_bytes.append(self.vocab[token])
        return b"".join(chunk_bytes).decode("utf-8", errors="replace")

    def save(self, path: str):
        """Saves vocab, merges, and special_tokens to file."""
        state = {
            "vocab": self.vocab,
            "merges": self.merges,
            "special_tokens": self.special_tokens,
        }
        torch.save(state, path)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Loads vocab, merges, and special_tokens from file."""
        state = torch.load(path)
        assert isinstance(state, dict), "state is not a dictionary"
        assert "vocab" in state, "vocab is not in state"
        assert "merges" in state, "merges is not in state"
        assert "special_tokens" in state, "special_tokens is not in state"
        return cls(
            vocab=state["vocab"],
            merges=state["merges"],
            special_tokens=state["special_tokens"],
        )

    def _pre_tokenize(
        self,
        text: str | None = None,
        texts: Iterator[str] | None = None,
        special_tokens: tuple[str, ...] = ("<|endoftext|>",),
        verbose: bool = False,
    ) -> dict[tuple[bytes, ...], int]:
        """Pre-tokenization step: split the text into chunks and count occurrences.

        Pre-tokenization takes long time and multiprocessing does not work in Colab
        so I just run pre-tokenization and save it to file.

        Args:
            text: Text containing all the documents. If None, `texts` is used instead.
            texts: An iterator of texts to be used for pre-tokenization. If None,
                `texts` is used instead.
            special_tokens: A tuple of strings that should not be split.
            verbose: Whether to print debug information.

        Returns:
            A dictionary mapping each unique chunk (as a tuple of bytes) to its
            frequency count.
        """
        assert text or texts, "Either text or texts should be provided"
        assert not (text and texts), "Only one of text or texts should be provided"
        # This step could be possibly parallelized. But multiprocessing in Colab is
        # poorly supported.

        def _pre_tokenize_text(
            text: str,
            chunk_counts: dict[tuple[bytes, ...], int],
            special_tokens_split_pat: str | None,
            verbose: bool = False,
        ):
            texts = [text]
            if special_tokens_split_pat != None:
                special_tokens_split_pat = "|".join(
                    re.escape(st) for st in special_tokens
                )
                texts = re.split(special_tokens_split_pat, text)
            for text in get_tqdm(texts, condition=verbose, desc="Pretokenizing"):
                for chunk in re.finditer(self.pat, text):
                    chunk_bytes = tuple(bytes([b]) for b in chunk.group().encode("utf-8"))
                    chunk_counts[chunk_bytes] = chunk_counts.get(chunk_bytes, 0) + 1

        special_tokens_split_pat = None
        if len(special_tokens) > 0:
            special_tokens_split_pat = "|".join(
                re.escape(st) for st in special_tokens
            )

        if text:
            chunk_counts = {}
            _pre_tokenize_text(text, chunk_counts, special_tokens_split_pat, verbose)
            return chunk_counts

        chunk_counts = {}
        for text in get_tqdm(texts, condition=verbose, desc="Pretokenizing"):
            _pre_tokenize_text(text, chunk_counts, special_tokens_split_pat, False)
        return chunk_counts
