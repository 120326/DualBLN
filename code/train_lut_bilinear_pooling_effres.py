import argparse
import os
import numpy as np
import math
import itertools
import time
import datetime
import sys
import torch
import random

from torch.utils.data import DataLoader
from torch.autograd import Variable
from models_x_lut_bilinear_pooling_effres import *
from datasets import *


def fix_randomness(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print("seed down")


fix_randomness(0)

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--data_path", type=str, default="/root", help="root of the datasets")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_smooth", type=float, default=0.0001,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--lambda_monotonicity", type=float, default=10.0,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--use_mask", type=bool, default=False,
                    help="whether to use the human region mask for weighted loss")
parser.add_argument("--lut_dim", type=int, default=36, help="dimension of lut")
parser.add_argument("--n_cpu", type=int, default=10, help="number of cpu threads to use during batch generation")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="epoch interval between model checkpoints")
parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/out_lut36_effresnet",
                    help="path to save or load model")
opt = parser.parse_args()

print(opt)

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

os.makedirs("%s" % opt.output_dir, exist_ok=True)

criterion_pixelwise = torch.nn.MSELoss()

LUT1 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT2 = Generator3DLUT_zero(dim=opt.lut_dim)
LUT3 = Generator3DLUT_zero(dim=opt.lut_dim)
LUT4 = Generator3DLUT_zero(dim=opt.lut_dim)
LUT5 = Generator3DLUT_zero(dim=opt.lut_dim)

classifier = Model()

TV3 = TV_3D(dim=opt.lut_dim)

trilinear_ = TrilinearInterpolation()

if cuda:
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    LUT5 = LUT5.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()
    TV3.cuda()
    TV3.weight_r = TV3.weight_r.type(Tensor)
    TV3.weight_g = TV3.weight_g.type(Tensor)
    TV3.weight_b = TV3.weight_b.type(Tensor)

if opt.epoch != 0:
    LUTs = torch.load("%s/LUTs_%d.pth" % (opt.output_dir, opt.epoch))
    LUT1.load_state_dict(LUTs["1"])
    LUT2.load_state_dict(LUTs["2"])
    LUT3.load_state_dict(LUTs["3"])
    LUT4.load_state_dict(LUTs["4"])
    LUT5.load_state_dict(LUTs["5"])

optimizer_G = torch.optim.Adam(
    itertools.chain(classifier.parameters(), LUT1.parameters(), LUT2.parameters(), LUT3.parameters(), LUT4.parameters(),
                    LUT5.parameters()), lr=opt.lr, betas=(opt.b1, opt.b2))

dataloader = DataLoader(
    ImageDataset_paper(opt.data_path, mode="train", use_mask=opt.use_mask),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.n_cpu,
)

psnr_dataloader = DataLoader(
    ImageDataset_paper(opt.data_path, mode="test", use_mask=opt.use_mask),
    batch_size=1,
    shuffle=False,
    num_workers=1,
)


def generator0(img):
    lut1 = LUT1.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut2 = LUT2.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut3 = LUT3.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut4 = LUT4.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut5 = LUT5.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    pred = classifier(img, lut1, lut2, lut3, lut4, lut5).squeeze()
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    gen_A3 = LUT3(img)
    gen_A4 = LUT4(img)
    gen_A5 = LUT5(img)

    combine_A = img.new(img.size())
    combine_A[0, :, :, :] = (
            pred[0] * gen_A1 + pred[1] * gen_A2 + pred[2] * gen_A3 + pred[3] * gen_A4 + pred[4] * gen_A5)
    weights_norm = torch.mean(pred ** 2)

    return combine_A, weights_norm


def generator_batch(img):
    lut1 = LUT1.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut2 = LUT2.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut3 = LUT3.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut4 = LUT4.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)
    lut5 = LUT5.state_dict()['LUT'].reshape(3, 216, 216).unsqueeze(dim=0)

    pred = classifier(img, lut1, lut2, lut3, lut4, lut5)
    if len(pred.shape) == 1:
        pred = pred.unsqueeze(0)
    gen_A1 = LUT1(img)
    gen_A2 = LUT2(img)
    gen_A3 = LUT3(img)
    gen_A4 = LUT4(img)
    gen_A5 = LUT5(img)

    combine_A = img.new(img.size())
    for i in range(img.size(0)):
        combine_A[i, :, :, :] = (
                pred[i, 0] * gen_A1[i, :, :, :] + pred[i, 1] * gen_A2[i, :, :, :] + pred[i, 2] * gen_A3[i, :, :,
                                                                                                 :] + pred[
                    i, 3] * gen_A4[i, :, :, :] + pred[i, 4] * gen_A5[i, :, :, :])
    weights_norm = torch.mean(pred ** 2)

    return combine_A, weights_norm


def calculate_psnr():
    classifier.eval()
    sum_psnr = 0
    for i, batch in enumerate(psnr_dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        fake_B, x = generator0(real_A)
        fake_B = torch.round(fake_B * 255)
        real_B = torch.round(real_B * 255)
        try:
            mse = criterion_pixelwise(fake_B, real_B)
        except:
            print(batch["input_name"])
        psnr = 10 * math.log10(255.0 * 255.0 / mse.item())
        sum_psnr += psnr

    return sum_psnr / len(psnr_dataloader)


prev_time = time.time()

max_psnr = 0
max_epoch = 0

for epoch in range(opt.epoch, opt.n_epochs):

    mse_sum = 0
    psnr_sum = 0

    classifier.train()

    for i, batch in enumerate(dataloader):

        real_A = Variable(batch["A_input"].type(Tensor))
        real_B = Variable(batch["A_exptC"].type(Tensor))

        if opt.use_mask:
            mask = Variable(batch["mask"].type(Tensor))

            mask = torch.sum(mask, 1)

            mask = mask.unsqueeze(1)

            weights = torch.ones_like(mask)
            weights[mask > 0] = 5

        optimizer_G.zero_grad()

        fake_B, weights_norm = generator_batch(real_A)

        if opt.use_mask:
            mse = criterion_pixelwise(fake_B * weights, real_B * weights)
        else:
            mse = criterion_pixelwise(fake_B, real_B)

        tv1, mn1 = TV3(LUT1)
        tv2, mn2 = TV3(LUT2)
        tv3, mn3 = TV3(LUT3)
        tv4, mn4 = TV3(LUT4)
        tv5, mn5 = TV3(LUT5)
        tv_cons = tv1 + tv2 + tv3 + tv4 + tv5
        mn_cons = mn1 + mn2 + mn3 + mn4 + mn5

        loss = mse + opt.lambda_smooth * (weights_norm + tv_cons) + opt.lambda_monotonicity * mn_cons

        psnr_sum += 10 * math.log10(1 / mse.item())

        mse_sum += mse.item()

        loss.backward()
        optimizer_G.step()

        batches_done = epoch * len(dataloader) + i

        batches_left = opt.n_epochs * len(dataloader) - batches_done

        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))

        prev_time = time.time()

        sys.stdout.write(
            "\r%s, [Epoch %d/%d] [Batch %d/%d] [psnr: %f, tv: %f, wnorm: %f, mn: %f] ETA: %s"
            % (
                opt.output_dir, epoch, opt.n_epochs, i, len(dataloader), psnr_sum / (i + 1), tv_cons, weights_norm,
                mn_cons,
                time_left,
            )
        )

    avg_psnr = calculate_psnr()

    if avg_psnr > max_psnr:
        max_psnr = avg_psnr
        max_epoch = epoch
    sys.stdout.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))

    if epoch % opt.checkpoint_interval == 0:
        LUTs = {"1": LUT1.state_dict(), "2": LUT2.state_dict(), "3": LUT3.state_dict(), "4": LUT4.state_dict(),
                "5": LUT5.state_dict()}

        torch.save(LUTs, "%s/LUTs_%d.pth" % (opt.output_dir, epoch))

        torch.save(classifier.state_dict(), "%s/classifier_%d.pth" % (opt.output_dir, epoch))

        file = open('%s/result.txt' % opt.output_dir, 'a')
        file.write(" [PSNR: %f] [max PSNR: %f, epoch: %d]\n" % (avg_psnr, max_psnr, max_epoch))

        file.close()
