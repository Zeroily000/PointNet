import torch


class TNet(torch.nn.Module):
    # Transformation Network T-Net
    def __init__(self, dim_feat):
        super(TNet, self).__init__()
        # input size: Bx3xN
        # output size: BxKxK, K = 3 or 64
        self.K = dim_feat

        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=dim_feat, out_channels=64, kernel_size=1, bias=True),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=64, out_channels=128, kernel_size=1, bias=True),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=128, out_channels=1024, kernel_size=1, bias=True),
            torch.nn.BatchNorm1d(num_features=1024),
            torch.nn.ReLU(inplace=True),

            # torch.nn.MaxPool1d(kernel_size=num_points)
        )  # Bx1024xN

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512, bias=True),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=512, out_features=256, bias=True),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(in_features=256, out_features=dim_feat ** 2, bias=True)
        )  # BxK^2

        # initialize with 0 so that the initial output is identity
        torch.nn.init.constant(self.classifier[-1].weight.data, 0)
        # torch.nn.init.constant(self.classifier[-1].bias.data, 0)
        self.classifier[-1].bias.data = torch.eye(dim_feat).view(1, -1).squeeze()
        # print self.classifier[-1].weight.data

    def forward(self, x):
        # x = self.features(x)  # Bx1024x1
        x = self.features(x).max(dim=-1, keepdim=True)[0]  # Bx1024x1
        x = torch.squeeze(x)  # Bx1024
        x = self.classifier(x)  # BxK^2

        # x should be initialized as identity matrix
        # x_init = torch.autograd.Variable(torch.eye(self.K))
        # if torch.cuda.is_available():
        #     x_init = x_init.cuda()
        # x = x.view(-1, self.K, self.K) + x_init  # BxKxK
        x = x.view(-1, self.K, self.K)  # BxKxK
        return x


class Feature3D(torch.nn.Module):
    # Feature Extractor
    def __init__(self):
        super(Feature3D, self).__init__()
        # input size: Bx3xN
        # output size: f1: Bx1024; f2: BxNx1088

        self.T1 = TNet(3)
        self.feature1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=3, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv1d(in_channels=64, out_channels=64, kernel_size=1),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(inplace=True)
        )  # Bx64xN

        self.T2 = TNet(64)
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

            # torch.nn.MaxPool1d(kernel_size=num_points)
        )  # Bx1024xN

    def forward(self, x):
        N = x.data.shape[-1]
        A1 = self.T1(x)  # Bx3x3
        x = torch.bmm(A1, x)  # Bx3xN
        x = self.feature1(x)  # Bx64xN

        A2 = self.T2(x)  # Bx64x64
        feat = torch.bmm(A2, x)  # Bx64xN
        f1 = self.feature2(feat).max(dim=-1, keepdim=True)[0]  # Bx1024x1
        f2 = torch.cat((feat, f1.repeat(1, 1, N)), dim=1)  # Bx1088xN

        return f1, f2, A1, A2


class PointNetClassification(torch.nn.Module):
    # PointNet Classification
    def __init__(self, num_classes):
        super(PointNetClassification, self).__init__()
        # input size: Bx3xN
        # output size: # Bxk
        self.features = Feature3D()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=1024, out_features=512, bias=True),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.ReLU(inplace=True),
            # torch.nn.Dropout(p=0.3),

            torch.nn.Linear(in_features=512, out_features=256, bias=True),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.3),

            torch.nn.Linear(in_features=256, out_features=num_classes, bias=True)
        )  # Bxk

    def forward(self, x):
        x, _, A1, A2 = self.features(x)  # Bx1024x1
        x = torch.squeeze(x, 2)
        x = self.classifier(x)  # Bxk
        return x, A1, A2


class PointNetSegmentation(torch.nn.Module):
    # PointNet Segmentation
    def __init__(self, num_classes):
        super(PointNetSegmentation, self).__init__()
        # input size: Bx9xN
        # output size: # (B*N)xm
        self.m = num_classes

        self.features = Feature3D()

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

            torch.nn.Conv1d(in_channels=128, out_channels=num_classes, kernel_size=1)
        )  # BxmxN

    def forward(self, x):
        _, x, A1, A2 = self.features(x)  # Bx1088xN
        x = self.classifier(x).transpose(1, 2)  # BxNxm
        x = x.contiguous().view(-1, self.m) # (B*N)xm
        return x, A1, A2


# test dimension
if __name__ == '__main__':
    B = 32
    N = 1000
    sim_data = torch.autograd.Variable(torch.rand(B, 3, N))
    tnet = TNet(3)
    clsnet = PointNetClassification(5)
    segnet = PointNetSegmentation(5)

    if torch.cuda.is_available():
        sim_data = sim_data.cuda()
        tnet = tnet.cuda()
        clsnet = clsnet.cuda()
        segnet = segnet.cuda()

    out = tnet(sim_data)
    print 'T:', out.size()

    out, _, _ = clsnet(sim_data)
    print 'Classifier:', out.size()

    out, _, _ = segnet(sim_data)
    print 'Segment:', out.size()
