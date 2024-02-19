import onnx
from onnx import helper, numpy_helper, TensorProto
import numpy as np
from utils import save_model

def generate_add_models(output_dir):
    # Test case: addition of scalars
    scalar_a = np.array(1.0, dtype=np.float32)
    scalar_b = np.array(2.0, dtype=np.float32)
    add_scalar_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Add',
                inputs=['scalar_a', 'scalar_b'],
                outputs=['sum'],
            ),
        ],
        name='AddScalarGraph',
        inputs=[
            helper.make_tensor_value_info('scalar_a', TensorProto.FLOAT, []),
            helper.make_tensor_value_info('scalar_b', TensorProto.FLOAT, []),
        ],
        outputs=[
            helper.make_tensor_value_info('sum', TensorProto.FLOAT, []),
        ],
    )
    add_scalar_model = helper.make_model(add_scalar_graph, producer_name='onnx-examples')
    save_model(add_scalar_model, output_dir, "add_scalar.onnx")

    # Test case: addition of matrices
    matrix_a = np.random.rand(2, 2).astype(np.float32)
    matrix_b = np.random.rand(2, 2).astype(np.float32)
    add_matrix_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Add',
                inputs=['matrix_a', 'matrix_b'],
                outputs=['sum'],
            ),
        ],
        name='AddMatrixGraph',
        inputs=[
            helper.make_tensor_value_info('matrix_a', TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info('matrix_b', TensorProto.FLOAT, [2, 2]),
        ],
        outputs=[
            helper.make_tensor_value_info('sum', TensorProto.FLOAT, [2, 2]),
        ],
    )
    add_matrix_model = helper.make_model(add_matrix_graph, producer_name='onnx-examples')
    save_model(add_matrix_model, output_dir, "add_matrix.onnx")

def generate_matmul_models(output_dir):
    # Test case: matrix multiplication 2D by 2D
    matrix_a = np.random.rand(2, 3).astype(np.float32)
    matrix_b = np.random.rand(3, 4).astype(np.float32)
    matmul_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'MatMul',
                inputs=['matrix_a', 'matrix_b'],
                outputs=['product'],
            ),
        ],
        name='MatMulGraph',
        inputs=[
            helper.make_tensor_value_info('matrix_a', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('matrix_b', TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info('product', TensorProto.FLOAT, [2, 4]),
        ],
    )
    matmul_model = helper.make_model(matmul_graph, producer_name='onnx-examples')
    save_model(matmul_model, output_dir, "matmul_2d_by_2d.onnx")

    # Test case: matrix multiplication 2D by 1D
    vector_b = np.random.rand(3).astype(np.float32)
    matmul_vector_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'MatMul',
                inputs=['matrix_a', 'vector_b'],
                outputs=['product'],
            ),
        ],
        name='MatMulVectorGraph',
        inputs=[
            helper.make_tensor_value_info('matrix_a', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('vector_b', TensorProto.FLOAT, [3]),
        ],
        outputs=[
            helper.make_tensor_value_info('product', TensorProto.FLOAT, [2]),
        ],
    )
    matmul_vector_model = helper.make_model(matmul_vector_graph, producer_name='onnx-examples')
    save_model(matmul_vector_model, output_dir, "matmul_2d_by_1d.onnx")

def generate_relu_model(output_dir):
    data = np.random.randn(3, 4, 5).astype(np.float32)
    relu_graph = helper.make_graph(
        nodes=[helper.make_node('Relu', inputs=['data'], outputs=['result'])],
        name='ReluGraph',
        inputs=[helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5])],
        outputs=[helper.make_tensor_value_info('result', TensorProto.FLOAT, [3, 4, 5])],
    )
    relu_model = helper.make_model(relu_graph, producer_name='onnx-examples')
    save_model(relu_model, output_dir, "relu.onnx")

def generate_concat_model(output_dir):
    data_1 = np.random.rand(2, 2).astype(np.float32)
    data_2 = np.random.rand(2, 2).astype(np.float32)
    concat_graph = helper.make_graph(
        nodes=[helper.make_node('Concat', inputs=['data_1', 'data_2'], outputs=['concat_result'], axis=0)],
        name='ConcatGraph',
        inputs=[
            helper.make_tensor_value_info('data_1', TensorProto.FLOAT, [2, 2]),
            helper.make_tensor_value_info('data_2', TensorProto.FLOAT, [2, 2]),
        ],
        outputs=[helper.make_tensor_value_info('concat_result', TensorProto.FLOAT, [4, 2])],
    )
    concat_model = helper.make_model(concat_graph, producer_name='onnx-examples')
    save_model(concat_model, output_dir, "concat.onnx")

def generate_transpose_model(output_dir):
    data = np.random.randn(2, 3, 4).astype(np.float32)
    transpose_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Transpose',
                inputs=['data'],
                outputs=['transposed'],
                perm=(1, 0, 2)  # Example: swap the first two dimensions
            ),
        ],
        name='TransposeGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4]),
        ],
        outputs=[
            helper.make_tensor_value_info('transposed', TensorProto.FLOAT, [3, 2, 4]),
        ],
    )
    transpose_model = helper.make_model(transpose_graph, producer_name='onnx-examples')
    save_model(transpose_model, output_dir, "transpose.onnx")

def generate_squeeze_model(output_dir):
    data = np.random.randn(1, 3, 4, 1).astype(np.float32)
    squeeze_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Squeeze',
                inputs=['data'],
                outputs=['squeezed'],
                axes=[0, 3]  # Remove the first and last dimension if they are single-dimensional
            ),
        ],
        name='SqueezeGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 3, 4, 1]),
        ],
        outputs=[
            helper.make_tensor_value_info('squeezed', TensorProto.FLOAT, [3, 4]),  # Expected shape after squeeze
        ],
    )
    squeeze_model = helper.make_model(squeeze_graph, producer_name='onnx-examples')
    save_model(squeeze_model, output_dir, "squeeze.onnx")

def generate_reshape_model(output_dir):
    data = np.random.randn(2, 3, 4).astype(np.float32)
    target_shape = np.array([4, 3, 2], dtype=np.int64)  # Example target shape, make sure the total elements match

    reshape_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Reshape',
                inputs=['data', 'shape'],
                outputs=['reshaped'],
            ),
        ],
        name='ReshapeGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4]),
            helper.make_tensor_value_info('shape', TensorProto.INT64, [3]),  # Shape is a 1D tensor with 3 elements in this case
        ],
        outputs=[
            helper.make_tensor_value_info('reshaped', TensorProto.FLOAT, [4, 3, 2]),  # Expected shape after Reshape
        ],
    )
    initializers = [
        numpy_helper.from_array(target_shape, name='shape'),
    ]
    reshape_model = helper.make_model(reshape_graph, producer_name='onnx-examples', initializers=initializers)
    save_model(reshape_model, output_dir, "reshape.onnx")

def generate_softmax_model(output_dir):
    data = np.random.randn(1, 5).astype(np.float32)  # Example: a batch with a single sample with 5 features

    softmax_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Softmax',
                inputs=['data'],
                outputs=['probabilities'],
                axis=1,  # Typically, softmax is applied along the feature axis
            ),
        ],
        name='SoftmaxGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 5]),
        ],
        outputs=[
            helper.make_tensor_value_info('probabilities', TensorProto.FLOAT, [1, 5]),  # Softmax normalized probabilities
        ],
    )
    softmax_model = helper.make_model(softmax_graph, producer_name='onnx-examples')
    save_model(softmax_model, output_dir, "softmax.onnx")

def generate_slice_model(output_dir):
    data = np.random.randn(3, 4, 5).astype(np.float32)  # Example 3D tensor
    starts = np.array([0, 1, 0], dtype=np.int64)  # Start indices for slicing along each axis
    ends = np.array([3, 3, 5], dtype=np.int64)  # End indices for slicing (exclusive)
    axes = np.array([0, 1, 2], dtype=np.int64)  # Axes to slice along
    steps = np.array([1, 1, 1], dtype=np.int64)  # Step along each axis

    slice_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Slice',
                inputs=['data', 'starts', 'ends', 'axes', 'steps'],
                outputs=['sliced'],
            ),
        ],
        name='SliceGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [3, 4, 5]),
            helper.make_tensor_value_info('starts', TensorProto.INT64, [3]),
            helper.make_tensor_value_info('ends', TensorProto.INT64, [3]),
            helper.make_tensor_value_info('axes', TensorProto.INT64, [3]),
            helper.make_tensor_value_info('steps', TensorProto.INT64, [3]),
        ],
        outputs=[helper.make_tensor_value_info('sliced', TensorProto.FLOAT, [3, 2, 5])],  # Example output shape after slicing
    )
    initializers = [
        numpy_helper.from_array(starts, name='starts'),
        numpy_helper.from_array(ends, name='ends'),
        numpy_helper.from_array(axes, name='axes'),
        numpy_helper.from_array(steps, name='steps'),
    ]
    slice_model = helper.make_model(slice_graph, producer_name='onnx-examples', initializers=initializers)
    save_model(slice_model, output_dir, "slice.onnx")

def generate_gather_model(output_dir):
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # Example 2D tensor
    indices = np.array([0, 1], dtype=np.int64)  # Indices to gather
    axis = 0  # Axis along which to gather

    gather_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Gather',
                inputs=['data', 'indices'],
                outputs=['gathered'],
                axis=axis,
            ),
        ],
        name='GatherGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3]),
            helper.make_tensor_value_info('indices', TensorProto.INT64, [2]),
        ],
        outputs=[helper.make_tensor_value_info('gathered', TensorProto.FLOAT, [2, 3])],  # Output shape depends on the indices and axis
    )
    initializers = [
        numpy_helper.from_array(indices, name='indices'),
    ]
    gather_model = helper.make_model(gather_graph, producer_name='onnx-examples', initializers=initializers)
    save_model(gather_model, output_dir, "gather.onnx")

def generate_mul_model(output_dir):
    data_a = np.random.randn(3, 4).astype(np.float32)
    data_b = np.random.randn(3, 4).astype(np.float32)

    mul_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Mul',
                inputs=['data_a', 'data_b'],
                outputs=['product'],
            ),
        ],
        name='MulGraph',
        inputs=[
            helper.make_tensor_value_info('data_a', TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info('data_b', TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info('product', TensorProto.FLOAT, [3, 4])],
    )
    mul_model = helper.make_model(mul_graph, producer_name='onnx-examples')
    save_model(mul_model, output_dir, "mul.onnx")

def generate_div_model(output_dir):
    data_a = np.random.randn(3, 4).astype(np.float32) + 1  # Avoid dividing by zero
    data_b = np.random.randn(3, 4).astype(np.float32) + 1

    div_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Div',
                inputs=['data_a', 'data_b'],
                outputs=['quotient'],
            ),
        ],
        name='DivGraph',
        inputs=[
            helper.make_tensor_value_info('data_a', TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info('data_b', TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info('quotient', TensorProto.FLOAT, [3, 4])],
    )
    div_model = helper.make_model(div_graph, producer_name='onnx-examples')
    save_model(div_model, output_dir, "div.onnx")

def generate_reduce_mean_model(output_dir):
    data = np.random.randn(2, 3, 4).astype(np.float32)

    reduce_mean_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'ReduceMean',
                inputs=['data'],
                outputs=['reduced'],
                axes=[1],  # Reduce along the second axis
                keepdims=0,
            ),
        ],
        name='ReduceMeanGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [2, 3, 4]),
        ],
        outputs=[helper.make_tensor_value_info('reduced', TensorProto.FLOAT, [2, 4])],  # Output shape after reduction
    )
    reduce_mean_model = helper.make_model(reduce_mean_graph, producer_name='onnx-examples')
    save_model(reduce_mean_model, output_dir, "reduce_mean.onnx")

def generate_batch_normalization_model(output_dir):
    data = np.random.randn(1, 2, 3, 4).astype(np.float32)  # Example input
    scale = np.random.rand(2).astype(np.float32)
    B = np.random.rand(2).astype(np.float32)
    mean = np.random.randn(2).astype(np.float32)
    var = np.random.rand(2).astype(np.float32)  # Variance must be positive

    batch_norm_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'BatchNormalization',
                inputs=['data', 'scale', 'B', 'mean', 'var'],
                outputs=['normalized'],
                epsilon=1e-5,  # Example epsilon
            ),
        ],
        name='BatchNormalizationGraph',
        inputs=[
            helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 2, 3, 4]),
            helper.make_tensor_value_info('scale', TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info('B', TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info('mean', TensorProto.FLOAT, [2]),
            helper.make_tensor_value_info('var', TensorProto.FLOAT, [2]),
        ],
        outputs=[helper.make_tensor_value_info('normalized', TensorProto.FLOAT, [1, 2, 3, 4])],
    )
    initializers = [
        numpy_helper.from_array(scale, name='scale'),
        numpy_helper.from_array(B, name='B'),
        numpy_helper.from_array(mean, name='mean'),
        numpy_helper.from_array(var, name='var'),
    ]
    batch_norm_model = helper.make_model(batch_norm_graph, producer_name='onnx-examples', initializers=initializers)
    save_model(batch_norm_model, output_dir, "batch_normalization.onnx")

def generate_max_model(output_dir):
    data_a = np.random.randn(3, 4).astype(np.float32)
    data_b = np.random.randn(3, 4).astype(np.float32)

    max_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Max',
                inputs=['data_a', 'data_b'],
                outputs=['maxed'],
            ),
        ],
        name='MaxGraph',
        inputs=[
            helper.make_tensor_value_info('data_a', TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info('data_b', TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info('maxed', TensorProto.FLOAT, [3, 4])],
    )
    max_model = helper.make_model(max_graph, producer_name='onnx-examples')
    save_model(max_model, output_dir, "max.onnx")

def generate_min_model(output_dir):
    data_a = np.random.randn(3, 4).astype(np.float32)
    data_b = np.random.randn(3, 4).astype(np.float32)

    min_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'Min',
                inputs=['data_a', 'data_b'],
                outputs=['minned'],
            ),
        ],
        name='MinGraph',
        inputs=[
            helper.make_tensor_value_info('data_a', TensorProto.FLOAT, [3, 4]),
            helper.make_tensor_value_info('data_b', TensorProto.FLOAT, [3, 4]),
        ],
        outputs=[helper.make_tensor_value_info('minned', TensorProto.FLOAT, [3, 4])],
    )
    min_model = helper.make_model(min_graph, producer_name='onnx-examples')
    save_model(min_model, output_dir, "min.onnx")

def generate_average_pool_model(output_dir):
    data = np.random.randn(1, 3, 32, 32).astype(np.float32)  # Example input: 1 sample, 3 channels, 32x32

    average_pool_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'AveragePool',
                inputs=['data'],
                outputs=['pooled'],
                kernel_shape=[2, 2],
                strides=[2, 2]
            ),
        ],
        name='AveragePoolGraph',
        inputs=[helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info('pooled', TensorProto.FLOAT, [1, 3, 16, 16])],  # Output shape after pooling
    )
    average_pool_model = helper.make_model(average_pool_graph, producer_name='onnx-examples')
    save_model(average_pool_model, output_dir, "average_pool.onnx")

def generate_global_average_pool_model(output_dir):
    data = np.random.randn(1, 3, 32, 32).astype(np.float32)  # Example input: 1 sample, 3 channels, 32x32

    global_average_pool_graph = helper.make_graph(
        nodes=[
            helper.make_node(
                'GlobalAveragePool',
                inputs=['data'],
                outputs=['global_pooled'],
            ),
        ],
        name='GlobalAveragePoolGraph',
        inputs=[helper.make_tensor_value_info('data', TensorProto.FLOAT, [1, 3, 32, 32])],
        outputs=[helper.make_tensor_value_info('global_pooled', TensorProto.FLOAT, [1, 3, 1, 1])],  # Output shape after pooling
    )
    global_average_pool_model = helper.make_model(global_average_pool_graph, producer_name='onnx-examples')
    save_model(global_average_pool_model, output_dir, "global_average_pool.onnx")

def ensure_dir_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def generate_test_models(operation, output_dir):
    ensure_dir_exists(output_directory)
    if operation.lower() == 'add':
        generate_add_models(output_dir)
    elif operation.lower() == 'matmul':
        generate_matmul_models(output_dir)
    elif operation.lower() == 'concat':
        generate_concat_model(output_dir)
    elif operation.lower() == 'relu':
        generate_relu_model(output_dir)
    elif operation.lower() == 'transpose':
        generate_transpose_model(output_dir)
    elif operation.lower() == 'squeeze':
        generate_squeeze_model(output_dir)
    elif operation.lower() == 'reshape':
        generate_reshape_model(output_dir)
    elif operation.lower() == 'softmax':
        generate_softmax_model(output_dir)
    elif operation.lower() == 'slice':
        generate_slice_model(output_dir)
    elif operation.lower() == 'gather':
        generate_gather_model(output_dir)
    elif operation.lower() == 'mul':
        generate_mul_model(output_dir)
    elif operation.lower() == 'div':
        generate_div_model(output_dir)
    elif operation.lower() == 'reduce_mean':
        generate_reduce_mean_model(output_dir)
    elif operation.lower() == 'batch_normalization':
        generate_batch_normalization_model(output_dir)
    elif operation.lower() == 'max':
        generate_max_normalization_model(output_dir)
    elif operation.lower() == 'min':
        generate_min_normalization_model(output_dir)
    elif operation.lower() == 'average_pool':
        generate_average_pool_normalization_model(output_dir)
    elif operation.lower() == 'global_average_pool':
        generate_global_average_pool_normalization_model(output_dir)
    else:
        print(f"Operation '{operation}' not supported yet.")