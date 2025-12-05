import numpy as np

def insert_column(array, column, index, expand_size=1):
    """
    Insert column at index. Returns array only if resizing was needed.
    
    Parameters:
    -----------
    array : np.ndarray
        The array to insert into
    column : np.ndarray
        The column to insert
    index : int
        The column index where to insert
    expand_size : int, optional
        Number of columns to add when resizing (default: 1)
        The new column will be placed at 'index', rest filled with zeros
    
    Returns:
    --------
    None if no resizing (array modified in-place)
    new_array if resizing was needed
    """
    
    if array.ndim == 1:
        if index < array.shape[0]:
            # Fast in-place replacement (no copying)
            array[index] = column
            return None
        else:
            # Need to resize - add expand_size columns
            n_cols = array.shape[0]
            # Create array with expand_size new columns
            new_cols = np.zeros(expand_size)
            
            # Place the column at the correct position
            col_position = index - n_cols
            new_cols[col_position] = column
            
            return np.concatenate([array, new_cols])
    else:
        if index < array.shape[1]:
            # Fast in-place replacement (no copying)
            array[:, index] = column
            return None
        else:
            # Need to resize - add expand_size columns
            n_rows = array.shape[0]
            n_cols = array.shape[1]
            
            # Create array with expand_size new columns
            new_cols = np.zeros((n_rows, expand_size))
            
            # Place the column at the correct position
            col_position = index - n_cols
            new_cols[:, col_position] = column
            
            return np.hstack([array, new_cols])