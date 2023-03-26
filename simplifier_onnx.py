import onnx
from onnxsim import simplify

import argparse


def simplify_model(model_path):
    model = onnx.load(model_path)
    model_simp, check = simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    output_path = model_path.replace('.onnx', '_simplified.onnx')
    onnx.save(model_simp, output_path)


def pase_args():
    """
    Parse the arguments
    Returns:
        The parsed arguments
    """
    desc = "Export the model to ONNX or PyTorch model format"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model_path', type=str, required=True, help='The path to the onnx model')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = pase_args()
    simplify_model(args.model_path)
    print('Simplified ONNX model successfully')
