import torch 
import os

current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(current_dir, "logs/go1_amp/exported/policies/policy_1.pt")
onnx_path = os.path.join(current_dir, "deploy_gazebo/go1_policy.onnx")

policy = torch.jit.load(model_path)

dummy_input = torch.randn(1, 42)
torch.onnx.export(
    policy,
    dummy_input,
    onnx_path,
    input_names=["obs"],
    output_names=["actions"],
    opset_version=11
)

# ==== verify the exported ONNX model ====
import onnx
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("ONNX model is valid and has been exported successfully to: ", onnx_path)