import polars as pl
from typing import Literal, Annotated, Union, IO

# Define a type for the file format
def open_file(
    file_path: Union[str, IO[bytes]],
    file_format: Annotated[
        Literal["csv", "parquet", "json", "xlsx"],
        "The format of the file to be opened. Supported formats are 'csv', 'parquet', 'json' and 'xlsx'."
    ]
) -> pl.DataFrame:
    """
    Opens a file and returns a Polars DataFrame.

    Args:
        file_path (Union[str, IO[bytes]]): The path to the file.
        file_format (str): The format of the file. Supported formats are 'csv', 'parquet', 'json' and 'xlsx'.

    Returns:
        pl.DataFrame: A Polars DataFrame containing the data from the file.
    """
    if file_format == "csv":
        return pl.read_csv(file_path)
    elif file_format == "parquet":
        return pl.read_parquet(file_path)
    elif file_format == "json":
        return pl.read_json(file_path)
    elif file_format == "xlsx":
        return pl.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_format}")