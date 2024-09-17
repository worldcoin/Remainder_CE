use ndarray::{s, Array, Array2};

/// Pads an `Array2<i64>` with extra rows of zeros.
///
/// # Arguments
/// * `arr` - The original 2D array.
/// * `new_num_rows` - The total number of rows after padding.
///
/// # Example
/// ```
/// use ndarray::{array, Array2};
/// use remainder::utils::array::pad_with_rows;
/// let arr = array![[1, 2], [3, 4], [5, 6]];
/// let padded_arr = pad_with_rows(arr, 5);
/// assert_eq!(padded_arr, array![[1, 2], [3, 4], [5, 6], [0, 0], [0, 0]]);
/// ```
///
pub fn pad_with_rows(arr: Array2<i64>, new_num_rows: usize) -> Array2<i64> {
    let (num_rows, num_cols) = arr.dim();
    assert!(new_num_rows >= num_rows);
    // Create a new array with the padded size
    let mut padded = Array2::<i64>::zeros((new_num_rows, num_cols));
    // Copy existing rows into the new array
    padded.slice_mut(s![0..num_rows, ..]).assign(&arr);
    padded
}
