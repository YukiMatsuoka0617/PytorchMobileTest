import torch
import torchvision

# resnetモデルを利用
model = torchvision.models.resnet18(pretrained=True)
# 推論modeに
model.eval()
example = torch.rand(1, 3, 224, 224)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("resnet.pt")