import argparse

import coremltools as ct
import torch
from pytorch_lightning import LightningModule
from torch.utils.mobile_optimizer import optimize_for_mobile

from AnimeGANv2 import AnimeGANv2


def export_to_onnx(model: LightningModule, input_sample):
    """
    Export the model to ONNX format
    Args:
        model: The model to be exported
        input_sample: The input sample to the model

    Returns:
        None
    """
    model.to_onnx('save_model/onnx/animeGan.onnx', input_sample=input_sample,
                  input_names=['input'], output_names=['output'])


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
        'input': {2: "height", 3: 'width'},
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


def export_to_torchscript_model(model: AnimeGANv2):
    """
    Export the model to TorchScript model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    traced_script_module = torch.jit.trace(model.generated, example_inputs=torch.randn(1, 3, 256, 256))
    traced_script_module.save('save_model/torchscript/animeGan.pt')


def export_to_coreml_model(model: AnimeGANv2):
    """
    Export the model to CoreML model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    model = model.generated
    model.eval()
    trace_model = torch.jit.trace(model, example_inputs=torch.randn(1, 3, 256, 256))

    scale = 1 / (0.226 * 255.0)
    bias = [- 0.485 / (0.229), - 0.456 / (0.224), - 0.406 / (0.225)]
    input_shape = ct.Shape(shape=(1, 3,
                                  ct.RangeDim(lower_bound=0, upper_bound=-1, default=256),
                                  ct.RangeDim(lower_bound=0, upper_bound=-1, default=256)))
    image_input = ct.ImageType(name="input",
                               shape=input_shape,
                               scale=scale, bias=bias)
    # input = ct.TensorType(name='input', shape=(1, 3,
    #                                                 ct.RangeDim(lower_bound=0, upper_bound=-1, default=256),
    #                                                 ct.RangeDim(lower_bound=0, upper_bound=-1, default=256)))

    mlmodel = ct.convert(trace_model, inputs=[image_input],
                         outputs=[ct.TensorType(name='output')],
                         debug=True)
    mlmodel.save('save_model/coreml/animeGan.mlmodel')


def export_to_torch_mobile_model(model: AnimeGANv2):
    """
    Export the model to Torch Mobile model format
    Args:
        model: The model to be exported

    Returns:
        None
    """
    model = model.generated
    model.eval()
    example_inputs = torch.randn(1, 3, 256, 256)
    traced_script_module = torch.jit.trace(model, example_inputs=example_inputs)
    optimized_model = optimize_for_mobile(traced_script_module, backend='metal')
    print(torch.jit.export_opnames(optimized_model))
    optimized_model._save_for_lite_interpreter('save_model/torch_mobile/animeGan_metal.pt')


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
    parser.add_argument('--torchscript', action='store_true', help='Export to TorchScript model format')
    parser.add_argument('--coreml', action='store_true', help='Export to CoreML model format')
    parser.add_argument('--torch_mobile', action='store_true', help='Export to Torch Mobile model format')
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
    if args.torchscript:
        export_to_torchscript_model(model)
        print('Export to TorchScript model format successfully')
    if args.coreml:
        export_to_coreml_model(model)
        print('Export to CoreML model format successfully')
    if args.torch_mobile:
        export_to_torch_mobile_model(model)
        print('Export to Torch Mobile model format successfully')
