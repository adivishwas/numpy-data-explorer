"""
================================================================================
                    NumPy Data Explorer - Complete Guide
================================================================================
This project demonstrates NumPy fundamentals for Data Science beginners.
It covers array operations, mathematical computations, reshaping, broadcasting,
file I/O, and performance comparisons.

Author: Data Science Intern
Purpose: Educational project demonstrating NumPy capabilities
================================================================================
"""

import numpy as np
import time
import os

# ============================================================================
# SECTION 1: NumPy Fundamentals - Array Creation, Indexing, and Slicing
# ============================================================================

print("\n" + "="*80)
print("SECTION 1: NumPy Fundamentals - Array Creation, Indexing, and Slicing")
print("="*80)

# ----------------------------------------------------------------------------
# 1.1 Array Creation using Different Methods
# ----------------------------------------------------------------------------

print("\n--- 1.1 Array Creation Methods ---\n")

# Method 1: From Python list
print("Method 1: Creating array from Python list")
python_list = [1, 2, 3, 4, 5]
arr_from_list = np.array(python_list)
print(f"Python list: {python_list}")
print(f"NumPy array: {arr_from_list}")
print(f"Array type: {type(arr_from_list)}")
print(f"Array shape: {arr_from_list.shape}")
print(f"Array dtype: {arr_from_list.dtype}\n")

# Method 2: Using np.arange() - similar to range() but returns array
print("Method 2: Using np.arange() - creates array with evenly spaced values")
arr_arange = np.arange(0, 10, 2)  # start, stop, step
print(f"np.arange(0, 10, 2): {arr_arange}\n")

# Method 3: Using np.linspace() - creates array with specified number of points
print("Method 3: Using np.linspace() - creates evenly spaced numbers")
arr_linspace = np.linspace(0, 10, 5)  # start, stop, num_points
print(f"np.linspace(0, 10, 5): {arr_linspace}\n")

# Method 4: Using np.zeros() - creates array filled with zeros
print("Method 4: Using np.zeros() - creates array of zeros")
arr_zeros = np.zeros(5)
print(f"np.zeros(5): {arr_zeros}\n")

# Method 5: Using np.ones() - creates array filled with ones
print("Method 5: Using np.ones() - creates array of ones")
arr_ones = np.ones(5)
print(f"np.ones(5): {arr_ones}\n")

# Method 6: Using np.full() - creates array filled with specific value
print("Method 6: Using np.full() - creates array with specific value")
arr_full = np.full(5, 7)
print(f"np.full(5, 7): {arr_full}\n")

# Method 7: Using np.random - creates array with random values
print("Method 7: Using np.random - creates array with random values")
arr_random = np.random.randint(1, 10, size=5)
print(f"np.random.randint(1, 10, size=5): {arr_random}\n")

# Method 8: Creating 2D arrays (matrices)
print("Method 8: Creating 2D arrays (matrices)")
arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D array:\n{arr_2d}")
print(f"Shape: {arr_2d.shape}  # (rows, columns)\n")

# Method 9: Creating 3D arrays
print("Method 9: Creating 3D arrays")
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(f"3D array:\n{arr_3d}")
print(f"Shape: {arr_3d.shape}  # (depth, rows, columns)\n")

# ----------------------------------------------------------------------------
# 1.2 Indexing and Slicing
# ----------------------------------------------------------------------------

print("\n--- 1.2 Indexing and Slicing ---\n")

# Create a sample array for demonstration
sample_arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print(f"Sample array: {sample_arr}\n")

# Indexing: Accessing single elements
print("Indexing - Accessing single elements:")
print(f"First element (index 0): {sample_arr[0]}")
print(f"Last element (index -1): {sample_arr[-1]}")
print(f"Third element (index 2): {sample_arr[2]}\n")

# Slicing: Accessing multiple elements
print("Slicing - Accessing multiple elements:")
print(f"First 3 elements [0:3]: {sample_arr[0:3]}")
print(f"Elements from index 2 to 5 [2:6]: {sample_arr[2:6]}")
print(f"Last 3 elements [-3:]: {sample_arr[-3:]}")
print(f"Every 2nd element [::2]: {sample_arr[::2]}")
print(f"Reverse array [::-1]: {sample_arr[::-1]}\n")

# 2D Array Indexing and Slicing
print("2D Array Indexing and Slicing:")
matrix = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(f"Matrix:\n{matrix}\n")
print(f"Element at row 1, column 2 [1, 2]: {matrix[1, 2]}")
print(f"First row [0, :]: {matrix[0, :]}")
print(f"Second column [:, 1]: {matrix[:, 1]}")
print(f"Submatrix [0:2, 1:3]:\n{matrix[0:2, 1:3]}\n")

# ============================================================================
# SECTION 2: Mathematical Operations on NumPy Arrays
# ============================================================================

print("\n" + "="*80)
print("SECTION 2: Mathematical Operations on NumPy Arrays")
print("="*80)

# Create sample arrays for mathematical operations
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])
print(f"\nArray 1: {arr1}")
print(f"Array 2: {arr2}\n")

# ----------------------------------------------------------------------------
# 2.1 Element-wise Operations
# ----------------------------------------------------------------------------

print("--- 2.1 Element-wise Operations ---\n")
print("Element-wise operations perform the operation on each corresponding element")

# Addition
print("Addition (element-wise):")
result_add = arr1 + arr2
print(f"{arr1} + {arr2} = {result_add}\n")

# Subtraction
print("Subtraction (element-wise):")
result_sub = arr2 - arr1
print(f"{arr2} - {arr1} = {result_sub}\n")

# Multiplication
print("Multiplication (element-wise):")
result_mul = arr1 * arr2
print(f"{arr1} * {arr2} = {result_mul}\n")

# Division
print("Division (element-wise):")
result_div = arr2 / arr1
print(f"{arr2} / {arr1} = {result_div}\n")

# Power
print("Power (element-wise):")
result_pow = arr1 ** 2
print(f"{arr1} ** 2 = {result_pow}\n")

# ----------------------------------------------------------------------------
# 2.2 Axis-wise Operations (Row-wise and Column-wise)
# ----------------------------------------------------------------------------

print("--- 2.2 Axis-wise Operations ---\n")

# Create a 2D array for axis operations
matrix_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"2D Matrix:\n{matrix_2d}\n")
print("In NumPy, axis=0 refers to rows (vertical), axis=1 refers to columns (horizontal)\n")

# Sum along axis
print("Sum along axis:")
print(f"Sum along axis=0 (column-wise, down each column): {np.sum(matrix_2d, axis=0)}")
print(f"Sum along axis=1 (row-wise, across each row): {np.sum(matrix_2d, axis=1)}\n")

# Mean along axis
print("Mean along axis:")
print(f"Mean along axis=0 (column-wise): {np.mean(matrix_2d, axis=0)}")
print(f"Mean along axis=1 (row-wise): {np.mean(matrix_2d, axis=1)}\n")

# Max along axis
print("Max along axis:")
print(f"Max along axis=0 (column-wise): {np.max(matrix_2d, axis=0)}")
print(f"Max along axis=1 (row-wise): {np.max(matrix_2d, axis=1)}\n")

# Min along axis
print("Min along axis:")
print(f"Min along axis=0 (column-wise): {np.min(matrix_2d, axis=0)}")
print(f"Min along axis=1 (row-wise): {np.min(matrix_2d, axis=1)}\n")

# ----------------------------------------------------------------------------
# 2.3 Statistical Operations
# ----------------------------------------------------------------------------

print("--- 2.3 Statistical Operations ---\n")

# Create a sample array for statistics
data = np.array([23, 45, 67, 34, 56, 78, 12, 89, 45, 67])
print(f"Sample data: {data}\n")

# Mean (Average)
mean_value = np.mean(data)
print(f"Mean (Average): {mean_value:.2f}")

# Median (Middle value)
median_value = np.median(data)
print(f"Median: {median_value:.2f}")

# Standard Deviation (Measure of spread)
std_value = np.std(data)
print(f"Standard Deviation: {std_value:.2f}")

# Variance
variance_value = np.var(data)
print(f"Variance: {variance_value:.2f}")

# Sum
sum_value = np.sum(data)
print(f"Sum: {sum_value}")

# Minimum and Maximum
min_value = np.min(data)
max_value = np.max(data)
print(f"Minimum: {min_value}")
print(f"Maximum: {max_value}")

# Percentiles
percentile_25 = np.percentile(data, 25)
percentile_75 = np.percentile(data, 75)
print(f"25th Percentile: {percentile_25:.2f}")
print(f"75th Percentile: {percentile_75:.2f}\n")

# ============================================================================
# SECTION 3: Reshaping and Broadcasting
# ============================================================================

print("\n" + "="*80)
print("SECTION 3: Reshaping and Broadcasting")
print("="*80)

# ----------------------------------------------------------------------------
# 3.1 Reshaping Arrays
# ----------------------------------------------------------------------------

print("\n--- 3.1 Reshaping Arrays ---\n")

# Create a 1D array
original_arr = np.arange(12)
print(f"Original 1D array: {original_arr}")
print(f"Shape: {original_arr.shape}\n")

# Reshape to 2D
reshaped_2d = original_arr.reshape(3, 4)
print("Reshaped to 2D (3 rows, 4 columns):")
print(reshaped_2d)
print(f"Shape: {reshaped_2d.shape}\n")

# Reshape to 3D
reshaped_3d = original_arr.reshape(2, 3, 2)
print("Reshaped to 3D (2 depth, 3 rows, 2 columns):")
print(reshaped_3d)
print(f"Shape: {reshaped_3d.shape}\n")

# Flatten - Convert multi-dimensional array to 1D
print("Flatten - Convert to 1D:")
flattened = reshaped_2d.flatten()
print(f"Flattened array: {flattened}")
print(f"Shape: {flattened.shape}\n")

# Transpose - Swap rows and columns
print("Transpose - Swap rows and columns:")
transposed = reshaped_2d.T
print(f"Original:\n{reshaped_2d}")
print(f"Transposed:\n{transposed}")
print(f"Original shape: {reshaped_2d.shape}, Transposed shape: {transposed.shape}\n")

# ----------------------------------------------------------------------------
# 3.2 Broadcasting
# ----------------------------------------------------------------------------

print("--- 3.2 Broadcasting ---\n")
print("Broadcasting allows NumPy to perform operations on arrays of different shapes")
print("by automatically expanding smaller arrays to match larger ones.\n")

# Example 1: Adding scalar to array
print("Example 1: Adding a scalar to an array")
arr_broadcast1 = np.array([1, 2, 3, 4, 5])
scalar = 10
result1 = arr_broadcast1 + scalar
print(f"Array: {arr_broadcast1}")
print(f"Scalar: {scalar}")
print(f"Result: {result1}")
print("Explanation: The scalar (10) is broadcasted to [10, 10, 10, 10, 10]\n")

# Example 2: Adding 1D array to 2D array
print("Example 2: Adding 1D array to 2D array")
matrix_broadcast = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
vector_broadcast = np.array([10, 20, 30])
result2 = matrix_broadcast + vector_broadcast
print(f"Matrix:\n{matrix_broadcast}")
print(f"Vector: {vector_broadcast}")
print(f"Result:\n{result2}")
print("Explanation: The vector [10, 20, 30] is broadcasted to each row\n")

# Example 3: Column-wise broadcasting
print("Example 3: Column-wise broadcasting")
matrix_broadcast2 = np.array([[1, 2, 3], [4, 5, 6]])
vector_col = np.array([[10], [20]])
result3 = matrix_broadcast2 + vector_col
print(f"Matrix:\n{matrix_broadcast2}")
print(f"Column vector:\n{vector_col}")
print(f"Result:\n{result3}")
print("Explanation: The column vector is broadcasted to each column\n")

# Example 4: Multiplication with broadcasting
print("Example 4: Multiplication with broadcasting")
arr_mult = np.array([[1, 2, 3], [4, 5, 6]])
multiplier = np.array([2, 3, 4])
result4 = arr_mult * multiplier
print(f"Array:\n{arr_mult}")
print(f"Multiplier: {multiplier}")
print(f"Result:\n{result4}\n")

# ----------------------------------------------------------------------------
# 3.3 Why Broadcasting Improves Efficiency
# ----------------------------------------------------------------------------

print("--- 3.3 Why Broadcasting Improves Efficiency ---\n")
print("Broadcasting improves efficiency because:")
print("1. No need to create full-size copies of smaller arrays")
print("2. Operations are performed in-place where possible")
print("3. Reduces memory usage significantly")
print("4. Enables vectorized operations that are faster than loops")
print("5. Leverages optimized C code under the hood\n")

# Demonstration: Broadcasting vs. Manual Loop
print("Demonstration: Broadcasting vs. Manual Loop")
large_array = np.random.rand(1000, 1000)
scalar_value = 5

# Broadcasting approach
start_time = time.time()
result_broadcast = large_array + scalar_value
broadcast_time = time.time() - start_time

# Manual loop approach (slower)
start_time = time.time()
result_loop = np.zeros_like(large_array)
for i in range(large_array.shape[0]):
    for j in range(large_array.shape[1]):
        result_loop[i, j] = large_array[i, j] + scalar_value
loop_time = time.time() - start_time

print(f"Broadcasting time: {broadcast_time:.6f} seconds")
print(f"Manual loop time: {loop_time:.6f} seconds")
print(f"Speedup: {loop_time/broadcast_time:.2f}x faster with broadcasting\n")

# ============================================================================
# SECTION 4: Save and Load Operations
# ============================================================================

print("\n" + "="*80)
print("SECTION 4: Save and Load Operations")
print("="*80)

# Create sample arrays to save
print("\n--- 4.1 Saving NumPy Arrays ---\n")

# Create arrays to save
array_to_save1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
array_to_save2 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
array_to_save3 = np.random.rand(5, 5)

print("Arrays to save:")
print(f"1D Array: {array_to_save1}")
print(f"2D Array:\n{array_to_save2}")
print(f"Random 5x5 Array:\n{array_to_save3}\n")

# Method 1: Using np.save() - saves single array as .npy file
print("Method 1: Using np.save() - saves single array")
np.save('saved_array_1d.npy', array_to_save1)
print("Saved: saved_array_1d.npy")

# Method 2: Using np.savez() - saves multiple arrays as .npz file
print("\nMethod 2: Using np.savez() - saves multiple arrays")
np.savez('saved_arrays.npz', arr1d=array_to_save1, arr2d=array_to_save2, arr_random=array_to_save3)
print("Saved: saved_arrays.npz (contains 3 arrays)")

# Method 3: Using np.savetxt() - saves as text file (human-readable)
print("\nMethod 3: Using np.savetxt() - saves as text file")
np.savetxt('saved_array_text.txt', array_to_save2, fmt='%d')
print("Saved: saved_array_text.txt\n")

# ----------------------------------------------------------------------------
# 4.2 Loading NumPy Arrays
# ----------------------------------------------------------------------------

print("--- 4.2 Loading NumPy Arrays ---\n")

# Load single array
print("Loading single array using np.load():")
loaded_array1 = np.load('saved_array_1d.npy')
print(f"Loaded array: {loaded_array1}")
print(f"Arrays match: {np.array_equal(array_to_save1, loaded_array1)}\n")

# Load multiple arrays
print("Loading multiple arrays using np.load():")
loaded_arrays = np.load('saved_arrays.npz')
loaded_arr1d = loaded_arrays['arr1d']
loaded_arr2d = loaded_arrays['arr2d']
loaded_arr_random = loaded_arrays['arr_random']

print(f"Loaded 1D array: {loaded_arr1d}")
print(f"Loaded 2D array:\n{loaded_arr2d}")
print(f"Loaded random array:\n{loaded_arr_random}\n")

# Verify correctness
print("Verifying correctness:")
print(f"1D arrays match: {np.array_equal(array_to_save1, loaded_arr1d)}")
print(f"2D arrays match: {np.array_equal(array_to_save2, loaded_arr2d)}")
print(f"Random arrays match: {np.array_equal(array_to_save3, loaded_arr_random)}\n")

# Load text file
print("Loading text file using np.loadtxt():")
loaded_text = np.loadtxt('saved_array_text.txt', dtype=int)
print(f"Loaded from text:\n{loaded_text}")
print(f"Arrays match: {np.array_equal(array_to_save2, loaded_text)}\n")

# ============================================================================
# SECTION 5: Performance Comparison - NumPy vs Python Lists
# ============================================================================

print("\n" + "="*80)
print("SECTION 5: Performance Comparison - NumPy vs Python Lists")
print("="*80)

print("\n--- 5.1 Why NumPy is Faster ---\n")
print("NumPy is faster than Python lists because:")
print("1. NumPy arrays are stored in contiguous memory blocks")
print("2. Operations are vectorized (performed on entire arrays at once)")
print("3. NumPy uses optimized C/C++ code under the hood")
print("4. Less overhead - fixed data types vs. Python's dynamic typing")
print("5. Better cache utilization due to memory locality\n")

# ----------------------------------------------------------------------------
# 5.2 Performance Test 1: Array Creation
# ----------------------------------------------------------------------------

print("--- 5.2 Performance Test 1: Array Creation ---\n")

size = 1000000

# Python list creation
start_time = time.time()
python_list = list(range(size))
python_list_time = time.time() - start_time

# NumPy array creation
start_time = time.time()
numpy_array = np.arange(size)
numpy_array_time = time.time() - start_time

print(f"Creating {size:,} elements:")
print(f"Python list time: {python_list_time:.6f} seconds")
print(f"NumPy array time: {numpy_array_time:.6f} seconds")
print(f"NumPy is {python_list_time/numpy_array_time:.2f}x faster\n")

# ----------------------------------------------------------------------------
# 5.3 Performance Test 2: Element-wise Addition
# ----------------------------------------------------------------------------

print("--- 5.3 Performance Test 2: Element-wise Addition ---\n")

size = 1000000

# Create arrays/lists
python_list1 = list(range(size))
python_list2 = list(range(size, size*2))
numpy_arr1 = np.arange(size)
numpy_arr2 = np.arange(size, size*2)

# Python list addition (using list comprehension)
start_time = time.time()
python_result = [python_list1[i] + python_list2[i] for i in range(size)]
python_add_time = time.time() - start_time

# NumPy array addition
start_time = time.time()
numpy_result = numpy_arr1 + numpy_arr2
numpy_add_time = time.time() - start_time

print(f"Adding {size:,} elements:")
print(f"Python list time: {python_add_time:.6f} seconds")
print(f"NumPy array time: {numpy_add_time:.6f} seconds")
print(f"NumPy is {python_add_time/numpy_add_time:.2f}x faster\n")

# ----------------------------------------------------------------------------
# 5.4 Performance Test 3: Element-wise Multiplication
# ----------------------------------------------------------------------------

print("--- 5.4 Performance Test 3: Element-wise Multiplication ---\n")

size = 1000000

# Create arrays/lists
python_list = list(range(size))
numpy_arr = np.arange(size)

# Python list multiplication
start_time = time.time()
python_result = [x * 2 for x in python_list]
python_mul_time = time.time() - start_time

# NumPy array multiplication
start_time = time.time()
numpy_result = numpy_arr * 2
numpy_mul_time = time.time() - start_time

print(f"Multiplying {size:,} elements by 2:")
print(f"Python list time: {python_mul_time:.6f} seconds")
print(f"NumPy array time: {numpy_mul_time:.6f} seconds")
print(f"NumPy is {python_mul_time/numpy_mul_time:.2f}x faster\n")

# ----------------------------------------------------------------------------
# 5.5 Performance Test 4: Sum Operation
# ----------------------------------------------------------------------------

print("--- 5.5 Performance Test 4: Sum Operation ---\n")

size = 1000000

# Create arrays/lists
python_list = list(range(size))
numpy_arr = np.arange(size)

# Python list sum
start_time = time.time()
python_sum = sum(python_list)
python_sum_time = time.time() - start_time

# NumPy array sum
start_time = time.time()
numpy_sum = np.sum(numpy_arr)
numpy_sum_time = time.time() - start_time

print(f"Summing {size:,} elements:")
print(f"Python list time: {python_sum_time:.6f} seconds")
print(f"NumPy array time: {numpy_sum_time:.6f} seconds")
print(f"NumPy is {python_sum_time/numpy_sum_time:.2f}x faster")
print(f"Python sum result: {python_sum}")
print(f"NumPy sum result: {numpy_sum}")
print(f"Results match: {python_sum == numpy_sum}\n")

# ----------------------------------------------------------------------------
# 5.6 Performance Test 5: Mathematical Operations
# ----------------------------------------------------------------------------

print("--- 5.6 Performance Test 5: Square Root Operation ---\n")

size = 1000000

# Create arrays/lists
python_list = [i**2 for i in range(size)]
numpy_arr = np.arange(size)**2

# Python list square root (using list comprehension)
start_time = time.time()
python_result = [x**0.5 for x in python_list]
python_sqrt_time = time.time() - start_time

# NumPy array square root
start_time = time.time()
numpy_result = np.sqrt(numpy_arr)
numpy_sqrt_time = time.time() - start_time

print(f"Computing square root of {size:,} elements:")
print(f"Python list time: {python_sqrt_time:.6f} seconds")
print(f"NumPy array time: {numpy_sqrt_time:.6f} seconds")
print(f"NumPy is {python_sqrt_time/numpy_sqrt_time:.2f}x faster\n")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThis project demonstrated:")
print("- NumPy array creation using various methods")
print("- Indexing and slicing operations")
print("- Element-wise and axis-wise mathematical operations")
print("- Statistical operations (mean, median, std, etc.)")
print("- Array reshaping, flattening, and transposing")
print("- Broadcasting and its efficiency benefits")
print("- Saving and loading NumPy arrays")
print("- Performance comparison showing NumPy's superiority over Python lists")
print("\nNumPy is essential for Data Science because it provides:")
print("- Fast numerical computations")
print("- Efficient memory usage")
print("- Vectorized operations")
print("- Foundation for pandas, scikit-learn, and other data science libraries")
print("\n" + "="*80)
print("Project Complete! Thank you for exploring NumPy with us!")
print("="*80 + "\n")