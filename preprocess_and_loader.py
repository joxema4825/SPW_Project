import numpy as np
import torch
import segmentation_models_pytorch as smp
import torch.nn as nn
import matplotlib.pyplot as plt
import nibabel as nib

class PretrainedUNet2D(nn.Module):
    def __init__(self, in_channels=4, num_classes=4, encoder_name='resnet34', pretrained=True):
        super().__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

def file_loader(flair_file, t1_file, t1ce_file, t2_file):
    flair_data = nib.load(flair_file).get_fdata()
    t1_data = nib.load(t1_file).get_fdata()
    t1ce_data = nib.load(t1ce_file).get_fdata()
    t2_data = nib.load(t2_file).get_fdata()

    return flair_data, t1_data, t1ce_data, t2_data

def normalize(nparray):
  nparraymean = np.mean(nparray[nparray > 0])
  nparraystd = np.std(nparray[nparray > 0])
  return (nparray - nparraymean)/nparraystd if nparraystd != 0 else nparray

def Preprocess(flair_data, t1_data, t1ce_data, t2_data):
    flair_data = normalize(flair_data)
    t1_data = normalize(t1_data)
    t2_data = normalize(t2_data)
    t1ce_data = normalize(t1ce_data)

    input_slices = []

    for j in range(0, 155):
        flair_slice = flair_data[:, :, j]
        t1_slice = t1_data[:, :, j]
        t1ce_slice = t1ce_data[:, :, j]
        t2_slice = t2_data[:, :, j]

        input_slice = np.stack([flair_slice, t1_slice, t1ce_slice, t2_slice], axis=-1)
        input_slices.append(input_slice)

    return input_slices

def Predictor(input_slices, model, device):
    pred_slices = []
    for i in range(155):
        x = input_slices[i]
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(x)
            pred_mask = torch.argmax(pred, dim=1).squeeze().cpu().numpy()
            pred_slices.append(pred_mask)

    return pred_slices

