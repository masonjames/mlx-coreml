import argparse
import os
import torch
import coremltools as ct

def load_model(model_path):
    # Load the model architecture (assuming it's in the same directory)
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path).split('.')[0]
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(model_name, os.path.join(model_dir, f"{model_name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Assuming the model class name is the same as the file name (capitalized)
    model_class = getattr(module, model_name.capitalize())
    
    # Initialize the model
    model = model_class()
    
    # Load the state dict
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model

def convert_to_coreml(model, input_shape):
    # Define example input shape
    example_input = torch.rand(*input_shape)
    
    # Convert the PyTorch model to Core ML
    traced_model = torch.jit.trace(model, example_input)
    coreml_model = ct.convert(
        traced_model, 
        inputs=[ct.TensorType(shape=input_shape)]
    )
    
    return coreml_model

def main(model_path, input_shape):
    # Load the PyTorch model
    model = load_model(model_path)
    
    # Convert to Core ML
    coreml_model = convert_to_coreml(model, input_shape)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(model_path), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the Core ML model
    output_path = os.path.join(output_dir, f"{os.path.basename(model_path).split('.')[0]}.mlmodel")
    coreml_model.save(output_path)
    print(f"Core ML model saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to Core ML")
    parser.add_argument("model_path", type=str, help="Path to the PyTorch model (.pt file)")
    parser.add_argument("--input_shape", type=int, nargs='+', default=[1, 3, 224, 224],
                        help="Input shape for the model (default: 1 3 224 224)")
    
    args = parser.parse_args()
    
    main(args.model_path, tuple(args.input_shape))
