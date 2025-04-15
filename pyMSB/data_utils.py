import struct
from typing import BinaryIO

def read_data_simple_format(file_path: str) -> list[int]:
    """
    Reader for data in a simple text file format as a single data column.

    """
    with open(file_path, "rb") as file:
        content = file.read().decode("utf-8")
    data = [int(line.strip()) for line in content.split("\n") if line]
    return data


def read_data_fulldata_format(file_path: str) -> list[int]:
    """
    Reader for data generated using Olomoucké spektráky řízené Cčkovou aplikací (autor: Milan Vůjtek, KEF).

    The data is stored in a tab-separated format with the first column being the channel number
    and the second column being the counts.
    """
    with open(file_path, "rb") as file:
        content = file.read().decode("utf-8")
    data = [int(line.split("\t")[1].strip()) for line in content.split("\n") if line]
    return data


def read_data_binary_ms_format(file_path: str) -> list[int]:
    """
    Reader for data generated using Olomoucké spektráky řízené přes LabView.

    Unpacks the data as 32-bit unsigned integers and swaps the byte order.
    """
    with open(file_path, "rb") as file:
        content = file.read()
    data = [struct.unpack(">I", content[i : i + 4])[0] for i in range(0, len(content), 4)]
    return data


def read_data(file_path: str) -> list[int]:
    """Read data from a file.
    
    Args:
    -----
        file_path (str): Path to the file.
        
    Returns:
    --------
        list[int]: List of integers read from the file.

    """
    if file_path.endswith(".txt") or file_path.endswith(".csv"):
        return read_data_simple_format(file_path)
    elif file_path.endswith(".fulldata"):
        return read_data_fulldata_format(file_path)
    elif file_path.endswith(".ms"):
        return read_data_binary_ms_format(file_path)
    else:
        raise ValueError("Unsupported file format. Only .txt, .csv, .fulldata and .ms files are allowed.")
