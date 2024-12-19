from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.nn as nn


class DeepLabFeatureExtractor(nn.Module):
    def __init__(self, model):
        super(DeepLabFeatureExtractor, self).__init__()
        self.backbone = model.backbone  # Encoder
        self.classifier = model.classifier  # Decoder

    def forward(self, x):
        encoder_features = []

        # Pass through the encoder layers
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        # Extract features at specific layers
        for name, layer in self.backbone.named_children():
            if "layer" in name:
                x = layer(x)
                if "layer2" in name:  # Add the output from layer2 to encoder_features
                    encoder_features.append(x)

        # Perform the decoder forward pass
        decoder_output = self.classifier(x)

        return decoder_output, encoder_features
