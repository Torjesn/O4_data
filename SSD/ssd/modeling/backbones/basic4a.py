import torch
from typing import Tuple, List
import torch.nn as nn

class BasicModel(torch.nn.Module):
    """
    This is a basic backbone for SSD.
    The feature extractor outputs a list of 6 feature maps, with the sizes:
    [shape(-1, output_channels[0], 38, 38),
     shape(-1, output_channels[1], 19, 19),
     shape(-1, output_channels[2], 10, 10),
     shape(-1, output_channels[3], 5, 5),
     shape(-1, output_channels[3], 3, 3),
     shape(-1, output_channels[4], 1, 1)]
    """
    def __init__(self,
            output_channels: List[int],
            image_channels: int,
            output_feature_sizes: List[Tuple[int]]):
        super().__init__()
        self.out_channels = output_channels
        self.output_feature_shape = output_feature_sizes
        
        self.layers = nn.ModuleDict({
            'layer_0': nn.Sequential(
                nn.Conv2d(
                in_channels=image_channels,
                out_channels= 32,
                kernel_size=3,
                stride=1,
                padding=1),

                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(
                in_channels=32,
                out_channels= 64,
                kernel_size=3,
                stride=1,
                padding=1),

                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1),

                nn.ReLU(),
                nn.Conv2d(
                in_channels=64,
                out_channels=output_channels[0],
                kernel_size=3,
                stride=2,
                padding=1),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
        })
        
        num_output_layers = len(self.out_channels)
        conv_1_filter = [64,128,256,128,128,128]
        conv_2_filter = self.out_channels
        conv_stride_1 = 1
        conv_stride_2 = 2
        padding_2 = 1
        
        for layer_n in range(1, num_output_layers):
            if layer_n == 5:
                conv_stride_2 = 1
                padding_2 = 0
            sec_n = nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(
                    in_channels = conv_2_filter[layer_n-1],
                    out_channels= conv_1_filter[layer_n],
                    kernel_size=3,
                    stride=conv_stride_1,
                    padding=1
                ),
                nn.ReLU(),
                nn.Conv2d(
                        in_channels = conv_1_filter[layer_n],
                    out_channels= conv_2_filter[layer_n],
                    kernel_size=3,
                    stride=conv_stride_2,
                    padding=padding_2
                ),
                nn.ReLU(),
                nn.Dropout(p=0.1)
            )
            self.layers.update(nn.ModuleDict({f"layer_{layer_n}": sec_n}))
            


    def forward(self, x):
        """
        The forward functiom should output features with shape:
            [shape(-1, output_channels[0], 38, 38),
            shape(-1, output_channels[1], 19, 19),
            shape(-1, output_channels[2], 10, 10),
            shape(-1, output_channels[3], 5, 5),
            shape(-1, output_channels[3], 3, 3),
            shape(-1, output_channels[4], 1, 1)]
        We have added assertion tests to check this, iteration through out_features,
        where out_features[0] should have the shape:
            shape(-1, output_channels[0], 38, 38),
        """
        out_features = []
        for _, layer in self.layers.items():
            x = layer(x)
            out_features.append(x)
        

        
        for idx, feature in enumerate(out_features):
            out_channel = self.out_channels[idx]
            h, w = self.output_feature_shape[idx]
            expected_shape = (out_channel, h, w)
            assert feature.shape[1:] == expected_shape, \
                f"Expected shape: {expected_shape}, got: {feature.shape[1:]} at output IDX: {idx}"
        assert len(out_features) == len(self.output_feature_shape),\
            f"Expected that the length of the outputted features to be: {len(self.output_feature_shape)}, but it was: {len(out_features)}"
        return tuple(out_features)

