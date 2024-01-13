import argparse
import model_generator

def main():
    parser = argparse.ArgumentParser(description='Generate ONNX test models for operations')
    parser.add_argument('operation', type=str, help='Name of the ONNX operation')
    parser.add_argument('--output_dir', type=str, default='./models', help='Directory to save generated models')
    
    args = parser.parse_args()
    
    model_generator.generate_test_models(args.operation, args.output_dir)

if __name__ == '__main__':
    main()