import torch
from AnimeGANv2 import AnimeGANv2
from pytorch_lightning.core.lightning import LightningModule
import argparse


def export_to_onnx(model: LightningModule, input_sample):
    """
    Export the model to ONNX format
    Args:
        model: The model to be exported
        input_sample: The input sample to the model

    Returns:
        None
    """
    model.to_onnx('save_model/onnx/animeGan.onnx', input_sample=input_sample)

def export_to_onnx_with_dynamic_input(model: LightningModule, input_sample):
    """
    Export the model to ONNX format with dynamic input
    Args:
        model: The model to be exported

    Returns:
        None
    """
    # 这么写表示NCHW都会变化
    dynamic_axes = {
        'input': {0: 'batch_size', 1: 'channel', 2: "height", 3: 'width'},
    }
    model.to_onnx('save_model/onnx/animeGan_dynamic.onnx', input_sample=input_sample, dynamic_axes=dynamic_axes,
                  input_names=['input'], output_names=['output'])


def export_to_pytorch_model(model: AnimeGANv2):
    """
    Export the model to PyTorch model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    torch.save(model.generated, 'save_model/pytorch/animeGan.pth')


def pase_args():
    """
    Parse the arguments
    Returns:
        The parsed arguments
    """
    desc = "Export the model to ONNX or PyTorch model format"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--checkpoint_path', type=str, required=True, help='The path to the checkpoint')
    parser.add_argument('--onnx', action='store_true', help='Export to ONNX format')
    parser.add_argument('--pytorch', action='store_true', help='Export to PyTorch model format')
    parser.add_argument('--dynamic', action='store_true', help='Export to ONNX format with dynamic input')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = pase_args()
    model = AnimeGANv2.load_from_checkpoint(args.checkpoint_path, strict=False)
    input_sample = torch.randn(1, 3, 256, 256)
    if args.onnx:
        export_to_onnx(model, input_sample)
        print('Export to ONNX format successfully')
    if args.pytorch:
        export_to_pytorch_model(model)
        print('Export to PyTorch model format successfully')
    if args.dynamic:
        export_to_onnx_with_dynamic_input(model, input_sample)
        print('Export to ONNX format with dynamic input successfully')
