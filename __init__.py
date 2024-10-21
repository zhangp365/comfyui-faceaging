# custom_nodes/comfyui_faceaging/__init__.py

import os
import warnings
import logging
import torch
import torchvision.transforms as transforms
import numpy as np
import dlib
from argparse import Namespace

from .datasets.augmentations import AgeTransformer 
from .utils.common import tensor2im  
from .scripts.align_all_parallel import align_face
from .models.psp import pSp

# Suppress specific warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set environment variable for CUDA architecture before importing torch-related modules
# os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'  # RTX A5000 값
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0'  # A100 값

# Initialize logging
logging.basicConfig(level=logging.WARNING)
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('torch').setLevel(logging.WARNING)

# Define model paths
CURRENT_DIR = 'models/face_aging/'
PREDICTOR_PATH = os.path.join(CURRENT_DIR, "shape_predictor_68_face_landmarks.dat")
MODEL_PATH = os.path.join(CURRENT_DIR, "sam_ffhq_aging.pt")

# Check and load shape predictor
if not os.path.exists(PREDICTOR_PATH):
    raise FileNotFoundError(f"Shape predictor not found at {PREDICTOR_PATH}")
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Define model paths dictionary
model_paths = {
    'psp_model': MODEL_PATH
}

# Check and load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location='cpu')  # weights_only 인자 제거
opts = ckpt.get('opts', {})
opts['checkpoint_path'] = MODEL_PATH
opts = Namespace(**opts)

# Initialize and load the pSp model
net = pSp(opts)
net.eval()
if torch.cuda.is_available():
    net.cuda()
logging.info('Model successfully loaded!')

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

class AgeTransformationNode:
    """
    ComfyUI Custom Node for Age Transformation using pSp.
    """
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "input_image": ("IMAGE",),
                "target_age": ("INT", {"default": 5, "min": 0, "max": 100})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "transform_age"
    CATEGORY = "Custom/FaceAging"

    def transform_age(self, input_image, target_age):
        """
        Transforms the input image to the specified target age.
        
        Parameters:
            input_image (PIL.Image or torch.Tensor): The input image to be aged.
            target_age (int): The desired target age.
        
        Returns:
            PIL.Image: The aged image.
        """

        input_image = input_image.squeeze(0)
        input_image = input_image.permute(2, 0, 1)
        input_image = input_image.detach().cpu()
        input_image = transforms.ToPILImage()(input_image)
        
        # Align the input image
        aligned_image = align_face(image=input_image, predictor=predictor)
        if aligned_image is None:
            raise ValueError("Face alignment failed.")

        # Preprocess the image
        input_tensor = transform(aligned_image)

        # Apply age transformation
        age_transformer = AgeTransformer(target_age=target_age)
        with torch.no_grad():
            input_image_age = age_transformer(input_tensor.cpu()).unsqueeze(0).to('cuda') # add batch dim
            result_tensor = net(input_image_age.float(), randomize_noise=False, resize=False)[0]
            result_image = tensor2im(result_tensor)
            # result_image.save('/workspace/qscar/faceAging/test.png')

        # Ensure the output tensor is on CPU and has shape [1, 3, 1024, 1024]
        result_tensor = torch.from_numpy(np.array(result_image).astype(np.float32) / 255.0).unsqueeze(0) # Add batch dimension if not present
        result_tensor = result_tensor.cpu()

        # logging.info(f'Age transformation successful: {result_tensor.shape}')
        return (result_tensor, )  # 단일 Tensor 객체 직접 반환
        

    def UI(self):
        return {
            "inputs": {
                "input_image": "Input Image",
                "target_age": "Target Age"
            },
            "outputs": {
                "Aged Image": "IMAGE"
            }
        }

NODE_CLASS_MAPPINGS = {
    "AgeTransformationNode": AgeTransformationNode
}
