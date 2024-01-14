import os

def save_model(model, output_dir, filename):
    # Save an ONNX model to a file
    filepath = os.path.join(output_dir, filename)
    print(f"Saving model to: {filepath}")
    onnx.save(model, filepath)