import os
import tempfile
from cs336_basics.utils import read_file_to_str_iterable, write_int_iterable_to_byte_file, read_byte_file_to_memmap, ROOT_PATH
import numpy as np


def test_read_file_to_str_iterable():
    path = ROOT_PATH / "tests" / "fixtures" / "tinystories_sample_5M.txt"
    str_iter = read_file_to_str_iterable(
        path,
        special_tokens=["<|endoftext|>"],
        buffer_size_bytes=1000,
    )
    got = "".join(str_iter)
    with open(path, "r", encoding="utf-8") as f:
        want = f.read()

    assert got == want

def test_write_and_read_int_iterable_to_memmap():
    with tempfile.TemporaryDirectory() as tmpdir:
        data_path = os.path.join(tmpdir, "data.bin")
        metadata_path = os.path.join(tmpdir, "metadata.json")
        # Data to write.
        arr = np.random.randint(0, 65536, size=10_000, dtype=np.uint16)
        write_int_iterable_to_byte_file(
            data_path,
            metadata_path,
            iter(arr),
            dtype=np.uint16,
            buffer_size_bytes=100,
        )
        # Read back with memmap.
        memmap_arr = read_byte_file_to_memmap(data_path, metadata_path)
        # Check type and values.
        assert isinstance(memmap_arr, np.memmap)
        assert np.array_equal(memmap_arr, arr)
