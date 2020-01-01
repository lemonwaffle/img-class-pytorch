import pretrainedmodels
import torch.nn as nn
import torch.nn.functional as F

from base import BaseModel


class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class ConvNet(BaseModel):
    """Generic convnet image classification model. 

    Uses pretrainedmodels from Cadene: https://github.com/Cadene/pretrained-models.pytorch
    """

    def __init__(self,
                 arch: str,
                 num_classes: int,
                 pretrained: bool = True,
                 freeze: bool = False,
                 ) -> None:
        """Initializes an image classification model.
        
        Parameters
        ----------
        arch : str
            Name of architecture.
        num_classes : int
            Number of classes to classify.
        pretrained : bool, optional
            Whether to use imagenet weights, by default True
        freeze : bool, optional
            Whether to freeze base arch (except final classifier layer), 
            by default False
        """

        super().__init__()

        # Initialize base architecture
        self.model = pretrainedmodels.__dict__[arch](
            pretrained='imagenet' if pretrained else None
        )

        # Freeze model (except for final classifier layer)
        if freeze:
            for params in self.model.parameters():
                params.requires_grad = False

        # Replace classifier layer
        self.model.last_linear = nn.Linear(
            self.model.last_linear.in_features, num_classes
        )

    def forward(self, x):
        return self.model(x)