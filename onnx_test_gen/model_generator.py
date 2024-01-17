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


def generate_test_models(operation, output_dir):
    if operation.lower() == 'add':
        generate_add_models(output_dir)
    elif operation.lower() == 'matmul':
        generate_matmul_models(output_dir)
    else:
        print(f"Operation '{operation}' not supported yet.")