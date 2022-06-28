import torch.nn as nn
import torchvision.models as models
import torch
import numpy as np
import trilinear
from resnet_lut import *
from efficientnet_lut import EfficientNet as create_model


def Bilinear_pooling(x, y):
    N = x.size()[0]
    x = x.view(N, 512, 1)
    y = y.view(N, 512, 1)
    x = torch.bmm(x, torch.transpose(y, 1, 2)) / 1 ** 2  # Bilinear
    assert x.size() == (N, 512, 512)
    x = x.view(N, 512 ** 2)
    x_sign = torch.sign(x)
    x = torch.sqrt(torch.abs(x + 1e-5))
    x = x * x_sign
    x = torch.nn.functional.normalize(x)

    return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')
        self.pad = nn.ZeroPad2d(4)
        self.net1 = create_model.from_pretrained('efficientnet-b0')
        self.net2 = Network_ResNet()
        self.conv = nn.Conv2d(1280, 512, kernel_size=1, stride=1, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, 5)
        self.out_fc = nn.Linear(262144, 5)

    def forward(self, img, lut1, lut2, lut3, lut4, lut5):
        img = self.upsample(img)
        result = self.net1(img)
        result = self.conv(result)
        weight_img = self.avgpool(result)
        weight_img = torch.flatten(weight_img, 1)
        weight_img = self.fc(weight_img)

        lut = torch.cat([lut1, lut2, lut3, lut4, lut5], dim=0)
        lut = self.pad(lut)
        lut = self.net2(lut)
        lut_out = weight_img[0, 0] * lut[0] + weight_img[0, 1] * lut[1] + weight_img[0, 2] * lut[2] + \
                  weight_img[0, 3] * lut[3] + weight_img[0, 4] * lut[4]
        lut_out = lut_out.unsqueeze(dim=0)
        if (weight_img.shape[0] != 1):
            for i in range(1, weight_img.shape[0]):
                t = weight_img[i, 0] * lut[0] + weight_img[i, 1] * lut[1] + weight_img[i, 2] * lut[2] + \
                    weight_img[i, 3] * lut[3] + weight_img[i, 4] * lut[4]
                t = t.unsqueeze(dim=0)
                lut_out = torch.cat([lut_out, t], dim=0)

        result = self.avgpool(result)
        lut_out = self.avgpool(lut_out)
        out = Bilinear_pooling(result, lut_out)
        out = self.out_fc(out)

        return out


def weights_init_normal_classifier(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.xavier_normal_(m.weight.data)

    elif classname.find("BatchNorm2d") != -1 or classname.find("InstanceNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ContextModulate(nn.Module):

    def __init__(self):
        super(ContextModulate, self).__init__()
        self.conv_1 = nn.Conv2d(27, 128, kernel_size=1, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(128, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        nn_Unfold = nn.Unfold(kernel_size=(3, 3), dilation=1, padding=1, stride=1)
        output_img = nn_Unfold(x)
        transform_img = output_img.view(x.shape[0], 27, x.shape[2], x.shape[3])
        out1 = self.relu(self.conv_1(transform_img))
        out2 = self.sigmoid(self.conv_2(out1))

        return out2


class resnet18_224(nn.Module):

    def __init__(self, out_dim=5, aug_test=False):
        super(resnet18_224, self).__init__()

        self.aug_test = aug_test
        net = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear')

        net.fc = nn.Linear(512, out_dim)
        self.model = net

    def forward(self, x):
        x = self.upsample(x)
        if self.aug_test:
            x = torch.cat((x, torch.flip(x, [3])), 0)
        f = self.model(x)

        return f


class Classifier(nn.Module):
    def __init__(self, in_channels=3):
        super(Classifier, self).__init__()

        self.model = nn.Sequential(
            nn.Upsample(size=(256, 256), mode='bilinear'),
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.InstanceNorm2d(16, affine=True),
            *discriminator_block(16, 32, normalization=True),
            *discriminator_block(32, 64, normalization=True),
            *discriminator_block(64, 128, normalization=True),
            *discriminator_block(128, 128),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 3, 8, padding=0),
        )

    def forward(self, img_input):
        return self.model(img_input)


class Generator3DLUT_identity(nn.Module):
    def __init__(self, dim=33):

        super(Generator3DLUT_identity, self).__init__()

        if dim == 33:
            file = open("IdentityLUT33.txt", 'r')
        elif dim == 36:
            file = open("/root/autodl-nas/IdentityLUT36.txt", 'r')
        elif dim == 64:
            file = open("IdentityLUT64.txt", 'r')
        lines = file.readlines()
        buffer = np.zeros((3, dim, dim, dim), dtype=np.float32)

        for i in range(0, dim):
            for j in range(0, dim):
                for k in range(0, dim):
                    n = i * dim * dim + j * dim + k

                    x = lines[n].split()

                    buffer[0, i, j, k] = float(x[0])
                    buffer[1, i, j, k] = float(x[1])
                    buffer[2, i, j, k] = float(x[2])

        self.LUT = nn.Parameter(torch.from_numpy(buffer).requires_grad_(True))

        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)
        return output


class Generator3DLUT_zero(nn.Module):
    def __init__(self, dim=33):
        super(Generator3DLUT_zero, self).__init__()

        self.LUT = nn.init.kaiming_normal_(torch.zeros(3, dim, dim, dim, dtype=torch.float), mode="fan_in",
                                           nonlinearity="relu")

        self.LUT = nn.Parameter(torch.tensor(self.LUT))

        self.TrilinearInterpolation = TrilinearInterpolation()

    def forward(self, x):
        _, output = self.TrilinearInterpolation(self.LUT, x)

        return output


class TrilinearInterpolationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut, x):

        x = x.contiguous()

        output = x.new(x.size())
        dim = lut.size()[-1]
        shift = dim ** 3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)

        if batch == 1:
            assert 1 == trilinear.forward(lut, x, output, dim, shift, binsize, W, H, batch)
        elif batch > 1:
            output = output.permute(1, 0, 2, 3).contiguous()
            assert 1 == trilinear.forward(lut, x.permute(1, 0, 2, 3).contiguous(), output, dim, shift, binsize, W, H,
                                          batch)

            output = output.permute(1, 0, 2, 3).contiguous()

        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]

        ctx.save_for_backward(*variables)

        return lut, output

    @staticmethod
    def backward(ctx, lut_grad, x_grad):
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])

        if batch == 1:
            assert 1 == trilinear.backward(x, x_grad, lut_grad, dim, shift, binsize, W, H, batch)
        elif batch > 1:
            assert 1 == trilinear.backward(x.permute(1, 0, 2, 3).contiguous(), x_grad.permute(1, 0, 2, 3).contiguous(),
                                           lut_grad, dim, shift, binsize, W, H, batch)
        return lut_grad, x_grad


class TrilinearInterpolation(torch.nn.Module):
    def __init__(self):
        super(TrilinearInterpolation, self).__init__()

    def forward(self, lut, x):
        return TrilinearInterpolationFunction.apply(lut, x)


class TV_3D(nn.Module):
    def __init__(self, dim=33):
        super(TV_3D, self).__init__()

        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0
        self.relu = torch.nn.ReLU()

    def forward(self, LUT):
        dif_r = LUT.LUT[:, :, :, :-1] - LUT.LUT[:, :, :, 1:]
        dif_g = LUT.LUT[:, :, :-1, :] - LUT.LUT[:, :, 1:, :]
        dif_b = LUT.LUT[:, :-1, :, :] - LUT.LUT[:, 1:, :, :]
        tv = torch.mean(torch.mul((dif_r ** 2), self.weight_r)) + torch.mean(
            torch.mul((dif_g ** 2), self.weight_g)) + torch.mean(torch.mul((dif_b ** 2), self.weight_b))

        mn = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + torch.mean(self.relu(dif_b))

        return tv, mn
