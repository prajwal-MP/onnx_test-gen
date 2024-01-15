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

def generate_test_models(operation, output_dir):
    if operation.lower() == 'add':
        generate_add_models(output_dir)
    else:
        print(f"Operation '{operation}' not supported yet.")