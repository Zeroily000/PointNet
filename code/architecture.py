import torch

class TNet(torch.nn.Module):
    # Transformation Network T-Net
    def __init__(self, num_points, dim_feat):
        super(TNet, self).__init__()
        # input size: Bx3xN
        # output size: BxKxK, K = 3 or 64
        self.K = dim_feat

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=dim_feat, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool1d(kernel_size=num_points)
        )  # Bx1024x1

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512, bias=True),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=512, out_features=256, bias=True),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=256, out_features=dim_feat**2, bias=True)
        )  # BxK^2

        # initialize with 0 so that the initial output is identity
        torch.nn.init.constant(self.classifier[-1].weight.data, 0)
        torch.nn.init.constant(self.classifier[-1].bias.data, 0)
        # print self.classifier[-1].weight.data

    def forward(self, x):
        x = self.features(x)  # Bx1024x1
        x = torch.squeeze(x)  # Bx1024
        x = self.classifier(x)  # BxK^2

        # x should be initialized as identity matrix
        x_init = torch.autograd.Variable(torch.eye(self.K))
        if torch.cuda.is_available():
            x_init = x_init.cuda()
        x = x.view(-1, self.K, self.K) + x_init  # BxKxK
        return x


class Feature3D(torch.nn.Module):
    # Feature Extractor
    def __init__(self, num_points):
        super(Feature3D, self).__init__()
        # input size: Bx3xN
        # output size: f1: Bx1024; f2: BxNx1088
        self.N = num_points

        self.T1 = TNet(num_points, 3)
        self.feature1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(inplace=True)
        )  # Bx64xN

        self.T2 = TNet(num_points, 64)
        self.feature2 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU(inplace=True),

            torch.nn.MaxPool1d(kernel_size=num_points)
        )  # Bx1024x1

    def forward(self, x):
        A1 = self.T1(x)  # Bx3x3
        x = torch.bmm(A1, x)  # Bx3xN
        x = self.feature1(x)  # Bx64xN

        A2 = self.T2(x)  # Bx64x64
        feat = torch.bmm(A2, x)  # Bx64xN
        f1 = self.feature2(feat)  # Bx1024x1
        f2 = torch.cat((feat, f1.repeat(1, 1, self.N)), dim=1)  # Bx1088xN

        return f1, f2


class PointNetClassification(torch.nn.Module):
    # PointNet Classification
    def __init__(self, num_points, num_classes):
        super(PointNetClassification, self).__init__()
        # input size: Bx3xN
        # output size: # Bxk
        self.features = Feature3D(num_points)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512, bias=True),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=512, out_features=256, bias=True),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.7),

            torch.nn.Linear(in_features=256, out_features=num_classes, bias=True)
        )  # Bxk

    def forward(self, x):
        x, _ = self.features(x)  # Bx1024x1
        x = torch.squeeze(x, 2)
        x = self.classifier(x)  # Bxk
        return x


class PointNetSegmentation(torch.nn.Module):
    # PointNet Segmentation
    def __init__(self, num_points, m):
        super(PointNetSegmentation, self).__init__()
        # input size: Bx3xN
        # output size: # BxNxm
        # torch.manual_seed(19260817)
        # torch.cuda.manual_seed_all(19260817)
        self.features = Feature3D(num_points)

        self.classifier = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=1088, out_channels=512, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=512, out_channels=256, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=256, out_channels=128, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=128, out_channels=m, kernel_size=1)
        )  # BxmxN

    def forward(self, x):
        _, x = self.features(x)  # Bx1088xN
        x = self.classifier(x)  # BxmxN
        return x.transpose(1, 2)


# test dimension
if __name__ == '__main__':
    B = 32
    N = 1000
    sim_data = torch.autograd.Variable(torch.rand(B, 3, N))
    tnet = TNet(N, 3)
    out = tnet(sim_data)
    print 'T:', out.size()

    clsnet = PointNetClassification(N, 5)
    out = clsnet(sim_data)
    print 'Classifier:', out.size()

    segnet = PointNetSegmentation(N, 5)
    out = segnet(sim_data)
    print 'Segment:', out.size()
