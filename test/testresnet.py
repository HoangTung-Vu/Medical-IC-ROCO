from torchvision.models import resnet50
import torch
import torch.nn as nn

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = resnet50(pretrained=False)
        encoder_layers = list(base_model.children())
        self.backbone = nn.Sequential(*encoder_layers[:9])
                        
    def forward(self, x):
        return self.backbone(x)

backbone = Backbone()
backbone.load_state_dict(torch.load("/home/hoangtungvum/CODE/MIC/test/ResNet50.pt", weights_only=True))

encoder = nn.Sequential(*list(backbone.backbone.children())[:-1]) 

dumb_image = torch.randn(1,3,224,224)
out = encoder(dumb_image)
print(out.shape)  # Expected output shape: (1, 512, 7, 7)