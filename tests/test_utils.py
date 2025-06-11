import pytest
import jax.numpy as jnp
import numpy as np
from jax import random

from thesis.utils import reshape_with_padding


def test_basic_reshape_no_padding():
    """Test basic reshape when sequence length is evenly divisible by num_branches."""
    x = jnp.ones((12, 3, 4))  # seq=12, batch=3, dim=4
    result = reshape_with_padding(x, num_branches=3)

    expected_shape = (3, 4, 3, 4)  # (num_branches, branch_len, batch, dim)
    assert result.shape == expected_shape

    # Verify no padding was added (all values should be 1)
    assert jnp.all(result == 1.0)


def test_reshape_with_padding():
    """Test reshape when padding is required."""
    x = jnp.ones((10, 2, 3))  # seq=10, batch=2, dim=3
    result = reshape_with_padding(x, num_branches=3)

    expected_shape = (3, 4, 2, 3)  # branch_len = ceil(10/3) = 4
    assert result.shape == expected_shape

    # Check that original data is preserved
    original_reshaped = result.reshape(-1, 2, 3)[:10]  # First 10 elements
    assert jnp.allclose(original_reshaped, x)

    # Check that padding is zeros
    padding_reshaped = result.reshape(-1, 2, 3)[10:]  # Last 2 elements
    assert jnp.allclose(padding_reshaped, 0.0)


def test_single_branch():
    """Test with num_branches=1 (no splitting)."""
    x = jnp.arange(24).reshape(6, 2, 2)
    result = reshape_with_padding(x, num_branches=1)

    expected_shape = (1, 6, 2, 2)
    assert result.shape == expected_shape

    # Should be equivalent to adding a dimension at the front
    expected = x[None, ...]
    assert jnp.allclose(result, expected)


def test_branches_equal_sequence_length():
    """Test when num_branches equals sequence length."""
    x = jnp.arange(20).reshape(5, 2, 2)
    result = reshape_with_padding(x, num_branches=5)

    expected_shape = (5, 1, 2, 2)  # Each branch has length 1
    assert result.shape == expected_shape

    # Verify data preservation
    for i in range(5):
        assert jnp.allclose(result[i, 0], x[i])


def test_branches_greater_than_sequence_length():
    """Test when num_branches > sequence length."""
    x = jnp.ones((3, 2, 4))
    result = reshape_with_padding(x, num_branches=5)

    expected_shape = (5, 1, 2, 4)  # branch_len = ceil(3/5) = 1
    assert result.shape == expected_shape

    # First 3 branches should have original data
    for i in range(3):
        assert jnp.allclose(result[i, 0], x[i])

    # Last 2 branches should be zeros (padding)
    for i in range(3, 5):
        assert jnp.allclose(result[i, 0], 0.0)


def test_different_data_types():
    """Test with different JAX array dtypes."""
    dtypes = [jnp.float32, jnp.float64, jnp.int32, jnp.complex64]

    for dtype in dtypes:
        x = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=dtype)
        result = reshape_with_padding(x, num_branches=2)

        assert result.dtype == dtype
        assert result.shape == (2, 1, 2, 2)


def test_preserve_values():
    """Test that values are preserved correctly during reshape."""
    # Create array with unique values to track positioning
    x = jnp.arange(24).reshape(4, 3, 2)
    result = reshape_with_padding(x, num_branches=2)

    # Should be shape (2, 2, 3, 2)
    assert result.shape == (2, 2, 3, 2)

    # Flatten and compare first 24 elements (original data)
    result_flat = result.reshape(-1)[:24]
    x_flat = x.reshape(-1)

    # The values should be rearranged but all present
    assert set(result_flat.tolist()) == set(x_flat.tolist())


def test_random_data_consistency():
    """Test with random data to ensure reshape is consistent."""
    key = random.PRNGKey(42)
    x = random.normal(key, shape=(7, 4, 3))

    result = reshape_with_padding(x, num_branches=3)

    # Should be shape (3, 3, 4, 3) with 2 padding elements
    assert result.shape == (3, 3, 4, 3)

    # Original data should be preserved in the first 7 positions
    result_flat = result.reshape(-1, 4, 3)[:7]
    assert jnp.allclose(result_flat, x)


def test_edge_case_empty_dimensions():
    """Test edge cases with small dimensions."""
    # Test with batch_size=1, dim=1
    x = jnp.ones((5, 1, 1))
    result = reshape_with_padding(x, num_branches=2)

    expected_shape = (2, 3, 1, 1)  # branch_len = ceil(5/2) = 3
    assert result.shape == expected_shape


# Error handling tests
def test_invalid_input_dimensions():
    """Test error handling for invalid input dimensions."""
    # Test 2D input
    x_2d = jnp.ones((5, 3))
    with pytest.raises(ValueError, match="Input must be 3D"):
        reshape_with_padding(x_2d, num_branches=2)

    # Test 4D input
    x_4d = jnp.ones((2, 3, 4, 5))
    with pytest.raises(ValueError, match="Input must be 3D"):
        reshape_with_padding(x_4d, num_branches=2)


def test_invalid_num_branches():
    """Test error handling for invalid num_branches values."""
    x = jnp.ones((5, 3, 4))

    # Test zero branches
    with pytest.raises(ValueError, match="num_branches must be positive"):
        reshape_with_padding(x, num_branches=0)

    # Test negative branches
    with pytest.raises(ValueError, match="num_branches must be positive"):
        reshape_with_padding(x, num_branches=-1)


def test_memory_layout_consistency():
    """Test that the function works with different memory layouts."""
    x = jnp.ones((6, 4, 3))

    # Test with C-order (default)
    result1 = reshape_with_padding(x, num_branches=2)

    # Test with F-order array
    x_f = jnp.asarray(np.asfortranarray(x))
    result2 = reshape_with_padding(x_f, num_branches=2)

    # Results should be the same regardless of memory layout
    assert jnp.allclose(result1, result2)


# Parametrized tests for comprehensive coverage
@pytest.mark.parametrize(
    "seq_len,num_branches", [(10, 3), (15, 4), (8, 2), (20, 5), (7, 3), (12, 6)]
)
def test_various_sizes(seq_len, num_branches):
    """Test various combinations of sequence lengths and branch numbers."""
    x = jnp.ones((seq_len, 2, 3))
    result = reshape_with_padding(x, num_branches=num_branches)

    expected_branch_len = (seq_len + num_branches - 1) // num_branches  # ceil division
    expected_shape = (num_branches, expected_branch_len, 2, 3)

    assert result.shape == expected_shape

    # Verify total elements after padding
    total_elements = num_branches * expected_branch_len * 2 * 3
    assert result.size == total_elements


@pytest.mark.parametrize("batch_size,dim_size", [(1, 1), (5, 10), (3, 7), (8, 4)])
def test_various_batch_dim_sizes(batch_size, dim_size):
    """Test various batch and dimension sizes."""
    x = jnp.ones((6, batch_size, dim_size))
    result = reshape_with_padding(x, num_branches=2)

    expected_shape = (2, 3, batch_size, dim_size)
    assert result.shape == expected_shape
