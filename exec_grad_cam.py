'''Script taking non-detected bird images and old model,
Outputting cropped regions around high-probability CAM areas
Author: Group 8 CS4245 - Jan Peter Simons'''

# Imports from pytorch_grad_cam
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from pytorch_grad_cam import GradCAM

# Standard imports
import argparse
import cv2
import numpy as np
import torch

# CUDA check
USE_CUDA = False
if torch.cuda.is_available():
    USE_CUDA = True

def get_args():
    parser = argparse.ArgumentParser()

    # Define to use cuda
    parser.add_argument('--use-cuda', action='store_true', default=USE_CUDA,
                        help='Use NVIDIA GPU acceleration')
    # Image path
    parser.add_argument(
        '--image-path',type=str, default='./examples/both.png',
                        help='Input image path')
    # Augmentation smoothener
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    # Reduce noise with eigen-smooth
    parser.add_argument(
        '--eigen_smooth', action='store_true',
                        help='Reduce noise by taking the first principle componenet'
                        'of cam_weights*activations')
    # Method = gradcam
    parser.add_argument(
        '--method',type=str,default='gradcam',
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')
    # Combine args
    args = parser.parse_args()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    
    return args

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))
    # Bring the channels to the first dimension, like in CNNs
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# ------------------------------------EXECUTE-----------------------------------
if __name__ == '__main__':
    
    method = GradCAM
    model = 
    target_layers = [model.backbone, model.layer4[-1] ]


    gradcam = method(model=model,
            target_layers=target_layers,
            use_cuda=args.use_cuda,
            reshape_transform=reshape_transform)