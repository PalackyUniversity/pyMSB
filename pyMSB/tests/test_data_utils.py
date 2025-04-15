import struct

from django.core.files.uploadedfile import SimpleUploadedFile

import pytest

from pyMSB.data_utils import (
    read_data,
    read_data_binary_ms_format,
    read_data_fulldata_format,
    read_data_simple_format,
)


def test_read_data_simple_format_with_valid_data():
    file = SimpleUploadedFile("test.txt", b"1\n2\n3\n")
    assert read_data_simple_format(file) == [1, 2, 3]


def test_read_data_fulldata_format_with_valid_data():
    file = SimpleUploadedFile("test.fulldata", b"0\t1\n1\t2\n2\t3\n")
    assert read_data_fulldata_format(file) == [1, 2, 3]


def test_read_data_binary_ms_format_with_valid_data():
    file = SimpleUploadedFile("test.ms", struct.pack(">I", 1) + struct.pack(">I", 2) + struct.pack(">I", 3))
    assert read_data_binary_ms_format(file) == [1, 2, 3]


def test_read_data_with_txt_file():
    file = SimpleUploadedFile("test.txt", b"1\n2\n3\n")
    assert read_data(file) == [1, 2, 3]


def test_read_data_with_fulldata_file():
    file = SimpleUploadedFile("test.fulldata", b"0\t1\n1\t2\n2\t3\n")
    assert read_data(file) == [1, 2, 3]


def test_read_data_with_ms_file():
    file = SimpleUploadedFile("test.ms", struct.pack(">I", 1) + struct.pack(">I", 2) + struct.pack(">I", 3))
    assert read_data(file) == [1, 2, 3]


def test_read_data_with_unsupported_file_format():
    file = SimpleUploadedFile("test.unsupported", b"")
    with pytest.raises(ValueError):
        read_data(file)
