import json
import os
import re
from typing import Iterator
import typing
import torch
import pathlib
import numpy as np
import sys


ROOT_PATH = (pathlib.Path(__file__).resolve().parent.parent)

# from google3.file.recordio.python import recordio
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


def read_file_to_str_iterable(
  path: str | os.PathLike,
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

def read_byte_file_to_memmap(path: str, metadata_path: str | None) -> np.memmap:
  """Reads a binary file of integers as a memory-mapped array.

  Args:
    path: The path to the binary file.
    metadata_path: The path to the metadata file.

  Returns:
    A memory-mapped array of integers.
  """
  if metadata_path is None:
    return np.load(path, mmap_mode="r")

  with open(metadata_path, "r") as f:
    metadata = json.load(f)
  dtype = np.dtype(metadata["dtype"])
  length = metadata["length"]

  # Memory-map the binary file.
  return np.memmap(path, dtype=dtype, mode="r", shape=(length,))


def save_checkpoint(
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
  iteration: int,
  out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
) -> None:
  """Saves model state, optimizer state, and iteration to a checkpoint file.

  Args:
    model: The model to save.
    optimizer: The optimizer to save.
    iteration: The current training iteration.
    out: Path or file-like object to serialize the model, optimizer, and
      iteration to.
  """
  out = pathlib.Path(out)
  torch.save(
    {
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict(),
      'iteration': iteration,
      'model_init_args': model.init_args() if hasattr(model, 'init_args') else {},
      'optimizer_init_args': optimizer.init_args() if hasattr(optimizer, 'init_args') else {},
    },
    out,
  )


def load_checkpoint(
  src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
  model_class: type[torch.nn.Module],
  optimizer_class: type[torch.optim.Optimizer],
  device: str | torch.device = 'cpu'
):
  """Loads model state, optimizer state, and iteration from a checkpoint file.

  Args:
    src: Path or file-like object to serialized checkpoint.
    model_class: The torch.nn.Module class to instantiate.
    optimizer_class: The optimizer class to instantiate.
    device: Device to load the model and optimizer to.

  Returns:
    (model, optimizer, iteration): The loaded model, optimizer, and iteration.
  """
  checkpoint = torch.load(src)
  if 'device' in checkpoint['model_init_args']:
    del checkpoint['model_init_args']['device']
  if 'device' in checkpoint['optimizer_init_args']:
    del checkpoint['optimizer_init_args']['device']
  model = model_class(device=device, **checkpoint['model_init_args'])
  optimizer = optimizer_class(
    params=model.parameters(),
    **checkpoint['optimizer_init_args'],
  )
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return model, optimizer, checkpoint['iteration']


def _load_checkpoint(
  src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
  model: torch.nn.Module,
  optimizer: torch.optim.Optimizer,
) -> int:
  """Loads model state, optimizer state, and iteration from a checkpoint file.

  Args:
    src: Path or file-like object to serialized checkpoint.
    model: The model to load the state into.
    optimizer: The optimizer to load the state into.
  Returns:
    The iteration number loaded from the checkpoint.
  """
  checkpoint = torch.load(src)
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  return checkpoint['iteration']