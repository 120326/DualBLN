import argparse
import time
import os
import torch
import torchvision
# import torchvision.utils as save_image
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable

from models_x_lut_bilinear_pooling_effres import *
from datasets_evaluation import *

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, default="/root/autodl-tmp/ppr_val/source", help="root of the datasets")
parser.add_argument("--gpu_id", type=str, default="0", help="gpu id")
parser.add_argument("--epoch", type=int, default=0, help="epoch to load")
parser.add_argument("--model_dir", type=str, default="/root/autodl-tmp/out_lut36_effresnet", help="path to save model")
parser.add_argument("--lut_dim", type=int, default=36, help="dimension of lut")
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

cuda = True if torch.cuda.is_available() else False

criterion_pixelwise = torch.nn.MSELoss()
# Initialize generator and discriminator
LUT1 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT2 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT3 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT4 = Generator3DLUT_identity(dim=opt.lut_dim)
LUT5 = Generator3DLUT_identity(dim=opt.lut_dim)
classifier = Model()
trilinear_ = TrilinearInterpolation()

if cuda:
    LUT1 = LUT1.cuda()
    LUT2 = LUT2.cuda()
    LUT3 = LUT3.cuda()
    LUT4 = LUT4.cuda()
    LUT5 = LUT5.cuda()
    classifier = classifier.cuda()
    criterion_pixelwise.cuda()

# Load pretrained models
LUTs = torch.load("%s/LUTs_%d.pth" % (opt.model_dir, opt.epoch))
LUT1.load_state_dict(LUTs["1"])
LUT2.load_state_dict(LUTs["2"])
LUT3.load_state_dict(LUTs["3"])
LUT4.load_state_dict(LUTs["4"])
LUT5.load_state_dict(LUTs["5"])

classifier.load_state_dict(torch.load("%s/classifier_%d.pth" % (opt.model_dir, opt.epoch)))
classifier.eval()



dataloader = DataLoader(
    ImageDataset_paper(opt.data_path),
    batch_size=1,
    shuffle=False,
    num_workers=10,
)

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def generator(img):
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

    return combine_A


def visualize_result():
    """Saves a generated sample from the validation set"""
    out_dir = "/root/autodl-tmp/saveimg"
    for i, batch in enumerate(dataloader):
        real_A = Variable(batch["A_input"].type(Tensor))
        img_name = batch["input_name"]
        fake_B = generator(real_A)
        save_image(fake_B, os.path.join(out_dir, "%s.png" % (img_name[0][:-4])), nrow=1, normalize=False)


with torch.no_grad():
    visualize_result()
