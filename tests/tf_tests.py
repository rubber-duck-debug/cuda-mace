import tensorflow as tf

def index_add(tensor, dim, index, values):
    """
    Adds values to tensor at the specified indices along the given dimension.
    
    Args:
        tensor: The input tensor.
        dim: The dimension along which to index.
        index: The indices where values will be added.
        values: The values to add to the tensor.
        
    Returns:
        A new tensor with values added at the specified indices along the given dimension.
    """
    # Create a range of indices along the specified dimension
    indices = tf.range(tf.shape(tensor)[dim], dtype=index.dtype)

    # Create a boolean mask to select the indices to be updated
    mask = tf.reduce_any(tf.equal(tf.expand_dims(indices, axis=-1), tf.expand_dims(index, axis=0)), axis=-1)

    # Use tf.where to update the selected indices with values
    updated_tensor = tf.where(tf.expand_dims(mask, axis=-1), tensor + tf.expand_dims(values, axis=1), tensor)

    return updated_tensor

# Example usage
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=tf.float32)
index = tf.constant([0, 2, 1])
values = tf.constant([10, 20, 30], dtype=tf.float32)
dim = 0

result = index_add(tensor, dim, index, values)

print(result.numpy())