"""Utility functions."""

import json
import re
from typing import Iterator, Protocol
import tqdm
import pathlib
import numpy as np
import sys


ROOT_PATH = (pathlib.Path(__file__).resolve().parent.parent)

# from google3.file.recordio.python import recordio


def get_tqdm(iterable, condition=True, **kwargs):
  if condition:
    return tqdm.tqdm(iterable, **kwargs)
  return iterable


# def load_data(
#     path: str,
#     num_records: int | None = None,
#     verbose: bool = False
# ) -> Iterator[str]:
#   """Loads data from a RecordIO file.

#   Args:
#     path: The path to the RecordIO file.
#     num_records: The maximum number of records to load. If None, all records
#       are loaded.
#     verbose: Whether to display a progress bar.

#   Yields:
#     Decoded records as UTF-8 strings.
#   """
#   with recordio.RecordReader(path) as rr:
#     for i, buf in enumerate(
#         get_tqdm(rr, condition=verbose, desc="Loading data")
#     ):
#       if num_records is not None and i >= num_records:
#         break
#       yield buf.decode("utf-8")


def concatenate_documents(data: list[str], special_token="<|endoftext|>"):
  return special_token.join(data)

def read_file_to_str_iterable(
  path: str,
  special_tokens=["<|endoftext|>"],
  buffer_size_bytes: int=10_000_000,
) -> Iterator[str]:
  """Yields texts from a file.
  
  Each yielded text must end with a special token and will be of size >=
  buffer_size_bytes.
  
  Args:
    path: The path to the input file.
    special_tokens: A list of special tokens to look for when searching for the
      end of the next text.
    buffer_size_bytes: The minimum size in bytes of the buffer before it yields
      the next text.
  """
  
  with open(path, "rb") as f:
    # Use a bytearray instead of a list of strings to save memory.
    # This is because a list of strings has a lot of overhead due to the
    # pointers to each string.
    buff = bytearray()
    for line in f:
      if sys.getsizeof(buff) < buffer_size_bytes:
        buff.extend(line)
        continue
      # Search for a line with the special token as we want to generate texts
      # ending with a special token.
      split = re.split(rf'(?<={"|".join(map(re.escape, special_tokens))})', line.decode("utf-8"), maxsplit=1)
      if len(split) == 1:
        buff.extend(line)
        continue
      assert len(split) == 2, "Split should have exactly two parts"
      buff.extend(split[0].encode("utf-8"))
      yield buff.decode("utf-8")
      buff = bytearray(split[1].encode("utf-8"))
    if len(buff) > 0:
      yield buff.decode("utf-8")

def write_int_iterable_to_byte_file(
  path: str, metadata_path: str,
  data: Iterator[int],
  dtype=np.uint16,
  buffer_size_bytes: int=10_000_000
) -> None:
  """Writes an iterable of integers to a binary file as dtype.

  Args:
    path: The path to the output file.
    metadata_path: The path to the metadata file, used to store dtype and total
      length, needed to properly read the file later.
    data: The iterable of integers to write.
    dtype: The numpy data type to use for writing.
    buffer_size_bytes: The number of bytes to buffer before flushing to disk.
  """
  
  # Clear the file first.
  with open(path, "wb") as f:
      pass  # Truncates the file to zero length

  # Open in the file in append mode.
  tot_len = 0
  with open(path, "ab") as f:
    buff = []
    curr_size = 0
    for x in data:
      tot_len += 1
      buff.append(x)
      curr_size += sys.getsizeof(x)
      if curr_size >= buffer_size_bytes:
        f.write(np.array(buff, dtype=dtype).tobytes())
        buff = []
        curr_size = 0
    # Write any remaining data in the buffer.
    if len(buff) > 0:
      f.write(np.array(buff, dtype=dtype).tobytes())
  
  # Write metadata.
  metadata = {
      "dtype": str(np.dtype(dtype)),
      "length": tot_len,
  }
  with open(metadata_path, "w") as f:
      json.dump(metadata, f)

def read_byte_file_to_memmap(path: str, metadata_path: str) -> np.memmap:
  """Reads a binary file of integers as a memory-mapped array.

  Args:
    path: The path to the binary file.
    metadata_path: The path to the metadata file.

  Returns:
    A memory-mapped array of integers.
  """
  with open(metadata_path, "r") as f:
    metadata = json.load(f)
  dtype = np.dtype(metadata["dtype"])
  length = metadata["length"]

  # Memory-map the binary file.
  return np.memmap(path, dtype=dtype, mode="r", shape=(length,))

class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

def compression_ratio(tokenizer: Tokenizer, text: str) -> float:
  """Computes the compression ratio (bytes/token) of a tokenizer on a given text.

  Args:
    tokenizer: The tokenizer to use.
    text: The text to encode.

  Returns:
    The compression ratio (bytes/token).
  """
  num_bytes = len(text.encode('utf-8'))
  num_tokens = len(tokenizer.encode(text))
  return num_bytes/num_tokens if num_tokens > 0 else float('inf')
