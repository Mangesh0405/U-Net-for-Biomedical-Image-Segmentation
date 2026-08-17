import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2 block for spatial feature extraction"""
    def __init__(self, in_channels: int, out_channels: int):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2, features: list = [64, 128, 256, 512]):
        """
        U-Net Architecture for Biomedical Image Segmentation (Ronneberger et al., 2015)
        
        Args:
            in_channels: Input image channels (1 for grayscale microscopy/CT, 3 for RGB).
            num_classes: Number of target segmentation classes (e.g., 2 for cell vs. background).
            features: Feature map channel depths across contracting resolution stages.
        """
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Contracting Path (Encoder)
        curr_in = in_channels
        for feature in features:
            self.downs.append(DoubleConv(curr_in, feature))
            curr_in = feature

        # Bottleneck (Deepest latent feature representation)
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # 2. Expansive Path (Decoder) with Transposed Convolutions
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        # 3. Final Output Layer (1x1 Conv to output pixel-level logits)
        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder forward pass
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Reverse skip connections for top-down spatial alignment during expansion
        skip_connections = skip_connections[::-1]

        # Decoder forward pass + Feature Map Concatenation
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # Upsampling via transposed conv
            skip_connection = skip_connections[idx // 2]

            # Adjust spatial dimensions if input dims are not perfectly divisible by 2^N
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True)

            # Concatenate skip features along channel dimension
            concat_x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_x)  # Double conv block

        return self.final_conv(x)


# ==========================================
# End-to-End Pipeline Usage Demonstration
# ==========================================
if __name__ == "__main__":
    # Hyperparameters
    IN_CHANNELS = 1       # Grayscale biomedical scan
    NUM_CLASSES = 2       # Pixel classification (0: Background, 1: Cell boundary)
    BATCH_SIZE = 4
    HEIGHT, WIDTH = 128, 128

    # Initialize UNet Model
    model = UNet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES)

    # Synthetic Batch of Input Images (Batch Size x Channels x Height x Width)
    dummy_input = torch.randn(BATCH_SIZE, IN_CHANNELS, HEIGHT, WIDTH)

    # Loss function and Optimizer setup
    criterion = nn.CrossEntropyLoss()  # Pixel-wise Cross Entropy
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Forward Pass
    logits = model(dummy_input)
    print("Output Logits Shape:", logits.shape)  # Expected: [4, 2, 128, 128]

    # Synthetic pixel-wise target mask (Batch Size x Height x Width)
    dummy_masks = torch.randint(low=0, high=NUM_CLASSES, size=(BATCH_SIZE, HEIGHT, WIDTH))

    # Calculate Pixel-wise Loss
    loss = criterion(logits, dummy_masks)

    # Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Training Step Successful. Pixel-wise Loss: {loss.item():.4f}")
