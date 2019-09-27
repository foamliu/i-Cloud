from torch import nn
from torchsummary import summary
from torchvision import models

from ps_config import device


class FrameDetectionModel(nn.Module):
    def __init__(self):
        super(FrameDetectionModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # print(resnet)
        # Remove linear layer (since we're not doing classification)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, 8)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images):
        x = self.resnet(images)  # [N, 2048, 1, 1]
        x = x.view(-1, 2048)  # [N, 2048]
        x = self.fc(x)
        x = self.sigmoid(x)  # [N, 8]
        return x


if __name__ == "__main__":
    from utils.utils import parse_args

    args = parse_args()
    model = FrameDetectionModel().to(device)
    # img=torch.Tensor(2,3,256,256)
    # out=model(img)
    # print(out.shape)
    summary(model, (3, 256, 256))
