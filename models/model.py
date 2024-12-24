from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DeepLabFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(DeepLabFeatureExtractor, self).__init__()
        self.backbone = model.backbone  # Encoder
        self.classifier = model.classifier  # Decoder
        # print(model)

    def forward(self, x):
        encoder_features = []
        decoder_features = []
        
        # Extract features from backbone (encoder)
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        for name, layer in self.backbone.named_children():
            if "layer" in name:
                x = layer(x)
                if "layer2" in name:
                    encoder_features.append(x)
        
        # Decoder forward pass
        decoder_output = self.classifier(x)
        decoder_features.append(decoder_output)
        
        return decoder_features, encoder_features