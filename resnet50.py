import torch
import torch.nn as nn

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, zero_init_residual=False):
        super(ResNet50, self).__init__()
        
        # Helper function for 1x1 convolutions
        def conv1x1(in_planes, out_planes, stride=1):
            """1x1 convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=1, 
                           stride=stride, bias=False)
        
        # Helper function for 3x3 convolutions
        def conv3x3(in_planes, out_planes, stride=1):
            """3x3 convolution with padding"""
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, 
                           stride=stride, padding=1, bias=False)
        
        # Bottleneck block for ResNet50
        class Bottleneck(nn.Module):
            expansion = 4

            def __init__(self, inplanes, planes, stride=1, downsample=None):
                super(Bottleneck, self).__init__()
                
                # First 1x1 convolution
                self.conv1 = conv1x1(inplanes, planes)
                self.bn1 = nn.BatchNorm2d(planes)
                
                # 3x3 convolution
                self.conv2 = conv3x3(planes, planes, stride)
                self.bn2 = nn.BatchNorm2d(planes)
                
                # Second 1x1 convolution
                self.conv3 = conv1x1(planes, planes * self.expansion)
                self.bn3 = nn.BatchNorm2d(planes * self.expansion)
                
                self.relu = nn.ReLU(inplace=True)
                self.downsample = downsample
                self.stride = stride

            def forward(self, x):
                identity = x

                # First bottleneck layer
                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                # Second bottleneck layer
                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                # Third bottleneck layer
                out = self.conv3(out)
                out = self.bn3(out)

                # Skip connection
                if self.downsample is not None:
                    identity = self.downsample(x)

                # Add identity
                out += identity
                out = self.relu(out)

                return out

        # Helper function to create ResNet layers
        def make_layer(inplanes, planes, blocks, stride=1):
            downsample = None

            # Create downsample layer if needed
            if stride != 1 or inplanes != planes * Bottleneck.expansion:
                downsample = nn.Sequential(
                    conv1x1(inplanes, planes * Bottleneck.expansion, stride),
                    nn.BatchNorm2d(planes * Bottleneck.expansion),
                )

            layers = []
            # First bottleneck block with potential downsampling
            layers.append(Bottleneck(inplanes, planes, stride, downsample))
            
            # Remaining bottleneck blocks
            inplanes = planes * Bottleneck.expansion
            for _ in range(1, blocks):
                layers.append(Bottleneck(inplanes, planes))

            return nn.Sequential(*layers)

        # Initial parameters
        self.inplanes = 64
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, 
                              padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet stages
        self.layer1 = make_layer(self.inplanes, 64, 3)  # 3 bottleneck blocks
        self.layer2 = make_layer(256, 128, 4, stride=2)  # 4 bottleneck blocks
        self.layer3 = make_layer(512, 256, 6, stride=2)  # 6 bottleneck blocks
        self.layer4 = make_layer(1024, 512, 3, stride=2)  # 3 bottleneck blocks

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                      nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet stages
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Global average pooling and final fully connected layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def get_layer_output(self, x, layer_name):
        """Helper function to get output of a specific layer"""
        layer_outputs = {}
        
        # Initial layers
        x = self.conv1(x)
        layer_outputs['conv1'] = x
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet stages
        x = self.layer1(x)
        layer_outputs['layer1'] = x
        x = self.layer2(x)
        layer_outputs['layer2'] = x
        x = self.layer3(x)
        layer_outputs['layer3'] = x
        x = self.layer4(x)
        layer_outputs['layer4'] = x
        
        # Final layers
        x = self.avgpool(x)
        layer_outputs['avgpool'] = x
        x = torch.flatten(x, 1)
        x = self.fc(x)
        layer_outputs['fc'] = x
        
        return layer_outputs[layer_name] if layer_name in layer_outputs else None

def test_resnet50():
    """Test function to verify the model architecture"""
    # Create model
    model = ResNet50()
    
    # Create sample input
    x = torch.randn(1, 3, 224, 224)
    
    # Forward pass
    output = model(x)
    
    # Print model summary
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test intermediate layer outputs
    layer_output = model.get_layer_output(x, 'layer3')
    print(f"Layer3 output shape: {layer_output.shape}")
    
    return model

if __name__ == "__main__":
    model = test_resnet50() 