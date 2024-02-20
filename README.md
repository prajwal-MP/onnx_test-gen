# ONNX Test Model Generator

This project provides a command-line tool for generating ONNX models for various operations to facilitate testing and experimentation with ONNX functionalities.

## Features

- Generate ONNX models for a range of operations including basic arithmetic, pooling, and normalization.
- Support for generating models with operations such as Add, Mul, Transpose, Squeeze, and more.
- Command-line interface for easy use and integration into testing workflows.

## Installation

First, clone this repository to your local machine:

```sh
git clone https://github.com/your-repo/onnx-test-gen.git
cd onnx-test-gen
```

Then, install the necessary Python dependencies:

```sh
pip install -r requirements.txt
```

## Usage
To generate ONNX models, run the main.py script from the command line with the operation name and optionally specify the output directory:

```sh
python main.py --operation Add --output_dir ./models
```

Replace Add with the operation you're interested in. See model_generator.py for a list of supported operation.