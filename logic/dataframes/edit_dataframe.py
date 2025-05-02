import polars as pl
from typing import Literal, Optional, OrderedDict

def edit_dataframe(
    df: pl.DataFrame,
    column: str,
    operation: Optional[Literal["add", "subtract", "multiply", "divide", "fill nan"]] = None,
    value: Optional[float] = None,
    new_column_name: Optional[str] = None
) -> pl.DataFrame:
    """
    Edit a DataFrame by performing an operation on a specified column.

    Args:
        df (pl.DataFrame): The input DataFrame.
        column (str): The name of the column to edit.
        operation (Literal["add", "subtract", "multiply", "divide", "fill nan"]): 
            The operation to perform. Supported operations: 
            'add', 'subtract', 'multiply', 'divide', 'fill nan'.
        value (float): The value to use in the operation.
        new_column_name (Optional[str]): The name of the new column. If None, the original column will be modified.

    Returns:
        pl.DataFrame: The modified DataFrame.
    """
    
    if new_column_name is None:
        new_column_name = column

    match operation:
        case "add":
            return df.with_columns(
                (pl.col(column) + value).alias(new_column_name)
            )
        case "subtract":
            return df.with_columns(
                (pl.col(column) - value).alias(new_column_name)
            )
        case "multiply":
            return df.with_columns(
                (pl.col(column) * value).alias(new_column_name)
            )
        case "divide":
            return df.with_columns(
                (pl.col(column) / value).alias(new_column_name)
            )
        case "fill nan":
            return df.with_columns(
                pl.when(pl.col(column).is_null())
                .then(value)
                .otherwise(pl.col(column))
                .alias(new_column_name)
            )
        case _:
            raise ValueError("Invalid operation. Choose from 'add', 'subtract', 'multiply', 'divide', or 'fill nan'.")