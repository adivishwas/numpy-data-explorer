# Output Examples

This file shows sample outputs from running `numpy_data_explorer.py`. 

## Quick Overview

The complete output is available in `sample_output.txt`. Below are key highlights:

## Section 1: Array Creation Examples

```
Method 1: Creating array from Python list
Python list: [1, 2, 3, 4, 5]
NumPy array: [1 2 3 4 5]
Array shape: (5,)
Array dtype: int64

Method 2: Using np.arange(0, 10, 2)
Result: [0 2 4 6 8]

Method 3: Using np.linspace(0, 10, 5)
Result: [ 0.   2.5  5.   7.5 10. ]
```

## Section 2: Mathematical Operations

### Element-wise Operations
```
Array 1: [1 2 3 4 5]
Array 2: [ 6  7  8  9 10]

Addition: [ 7  9 11 13 15]
Multiplication: [ 6 14 24 36 50]
Power (squared): [ 1  4  9 16 25]
```

### Statistical Operations
```
Sample data: [23 45 67 34 56 78 12 89 45 67]

Mean: 51.60
Median: 50.50
Standard Deviation: 23.18
Variance: 537.24
```

## Section 3: Broadcasting Example

```
Matrix:
[[1 2 3]
 [4 5 6]
 [7 8 9]]

Vector: [10 20 30]

Result (broadcasted):
[[11 22 33]
 [14 25 36]
 [17 28 39]]
```

**Broadcasting Performance:**
- Broadcasting time: ~0.005 seconds
- Manual loop time: ~0.6-0.8 seconds
- **Speedup: 100-200x faster!**

## Section 4: File I/O Verification

```
Saving arrays...
✓ saved_array_1d.npy
✓ saved_arrays.npz
✓ saved_array_text.txt

Loading and verification:
✓ 1D arrays match: True
✓ 2D arrays match: True
✓ Random arrays match: True
```

## Section 5: Performance Comparison Results

### Test Results (1,000,000 elements):

| Operation | Python List | NumPy Array | Speedup |
|-----------|-------------|-------------|---------|
| Array Creation | ~0.03s | ~0.003s | **8-12x faster** |
| Element Addition | ~0.18s | ~0.003s | **40-75x faster** |
| Element Multiplication | ~0.10s | ~0.003s | **22-30x faster** |
| Sum Operation | ~0.01s | ~0.001s | **7-9x faster** |
| Square Root | ~0.24s | ~0.006s | **40-50x faster** |

### Key Takeaway:
**NumPy is consistently 6-75x faster than Python lists for numerical operations!**

## Complete Output

For the full detailed output with all examples and explanations, see `sample_output.txt`.

---

*Note: Performance results may vary slightly based on your system specifications.*
