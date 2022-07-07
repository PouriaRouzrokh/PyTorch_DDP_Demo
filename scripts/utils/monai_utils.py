#-------------------------------------------------------------------------------
# Imports
#-------------------------------------------------------------------------------

# Standard built-in modules
import copy
import os
import shutil
from typing import Union
import warnings

# Third-party modules
import cv2
import monai as mn
import numpy as np
import pydicom
from PIL import Image as PILImage
from skimage.exposure import equalize_adapthist as sk_clahe
from skimage.util import invert as sk_invert
import timm
import torch

#-------------------------------------------------------------------------------
# Configurations
#-------------------------------------------------------------------------------

warnings.filterwarnings("ignore") 

#-------------------------------------------------------------------------------
# Custom MONAI transforms
#-------------------------------------------------------------------------------

#---------------------------------------
# - C: LoadCropD

class LoadCropD(mn.transforms.Transform):
    """A MONAI transform to load DICOM files using Pydicom.
    """
    def __init__(self, keys:list[str], dilation: int=0) -> None:
        """Initialize the transform.
        
        Args: 
            keys (list(str)): the input MONAI keys to the transformation class.
            dilation (int, optional): the dilation value for the bounding box.
                Defaults to 0.
        """
        super().__init__()
        self.keys=keys
        self.dilation = dilation

    def __call__(self, data: dict) -> dict:
        """Body of the transformation.
        
        Args:
            data (Dict): a dictionary of data to be transformed with MONAI.
        
        Returns:
            data_copy (Dict): the transformed data.
        """
        # Loading the image data.
        data_copy=copy.deepcopy(data)
        dcm = pydicom.dcmread(data_copy['image'])
        img = dcm.pixel_array
        if img.shape[-1] == 3:
            img = np.mean(img, axis=-1)
        if 'PhotometricInterpretation' in dcm.dir():
            if dcm.PhotometricInterpretation == 'MONOCHROME1':
                img = sk_invert(img)
                img -= np.min(img)
        
        # Cropping the image based on the bounding box.
        xmin, ymin, width, height = data_copy['crop_key']
        ymin = max(ymin - self.dilation, 0)
        ymax = min(ymin + height + self.dilation, img.shape[0])
        xmin = max(xmin - self.dilation, 0)
        xmax = min(xmin + width + self.dilation, img.shape[1])
        img = img[ymin:ymax, xmin:xmax]
        
        # Clipping the image data.
        percentile_low = np.percentile(img, 5)
        percentile_high = np.percentile(img, 95)
        img = np.clip(img, percentile_low, percentile_high)
        img = np.array(img, dtype='float32')
                
        # Adding the first channel and returning the image.
        img = np.expand_dims(img, axis=-1)
        data_copy["image"]=img
        return data_copy

#---------------------------------------
# - C: PadtoSquareD

class PadtoSquareD(mn.transforms.Transform):
    """A MONAI transform to pad an input image to square.
    """
    def __init__(self, keys:list[str]) -> None:
        """Initialize the transform.
        
        Args: 
            keys (list(str)): the input MONAI keys to the transformation class.
        """
        super().__init__()
        self.keys=keys

    def __call__(self, data: dict) -> dict:
        """Body of the transformation.
        
        Args:
            data (Dict): a dictionary of data to be transformed with MONAI.
        
        Returns:
            data_copy (Dict): the transformed data.
        """
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy().squeeze()
                height, width = img.shape
                if height < width: 
                    padded_img = np.zeros((width, width))
                    delta = (width - height) // 2
                    padded_img[delta:height+delta, :] = img
                    img = padded_img
                elif height > width:
                    padded_img = np.zeros((height, height))
                    delta = (height - width) // 2
                    padded_img[:, delta:width+delta] = img
                    img = padded_img
                data_copy[key]=np.expand_dims(img, axis=0)
        return data_copy
    
#---------------------------------------
# - C: CLAHED

class CLAHED(mn.transforms.Transform):
    """A MONAI transform to Apply Contrast Limited Adaptive Histogram 
    Equalization (CLAHE).
    """
    def __init__(self, keys:list[str]) -> None:
        """Initialize the transform.
        
        Args: 
            keys (list(str)): the input MONAI keys to the transformation class.
        """
        super().__init__()
        self.keys=keys

    def __call__(self, data: dict) -> dict:
        """Body of the transformation.
        
        Args:
            data (Dict): a dictionary of data to be transformed with MONAI.
        
        Returns:
            data_copy (Dict): the transformed data.
        """
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy()
                
                # Making 8bit and Applying the CLAHE algorithm.
                if img.dtype != np.dtype('uint8'):
                    img = img / np.max(img)
                    img = (img * 255).astype('uint8')
                img = sk_clahe(img, clip_limit=0.011)
                
                # Converting back to float32.
                img = (img / img.max()).astype(np.float32)
                data_copy[key]=img
        
        return data_copy

#---------------------------------------
# - C: EnsureGrayscaleD

class EnsureGrayscaleD(mn.transforms.Transform):
    """A MONAI transform to ensure that all images have 1 channel.
    """
    def __init__(self, keys:list[str]) -> None:
        """Initialize the transform.
        
        Args: 
            keys (list(str)): the input MONAI keys to the transformation class.
        """
        super().__init__()
        self.keys=keys

    def __call__(self, data: dict, add_1st_channel=False) -> dict:
        """Body of the transformation.
        
        Args:
            data (Dict): a dictionary of data to be transformed with MONAI.
        
        Returns:
            data_copy (Dict): the transformed data.
        """
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img = data[key].copy()
                if len(img.shape) == 3:
                    if img.shape[0] in [1, 3, 4]: # Channel First
                        img = np.mean(img, axis=0)
                    elif img.shape[-1] in [1, 3, 4]: # Channel Last
                        img = np.mean(img, axis=-1)
                elif not len(img.shape) == 2:
                    raise ValueError('Image shape is not 2D or 3D.')
                if add_1st_channel:
                    img = np.expand_dims(img,axis=0)
                data_copy[key] = img
        return data_copy
    
#---------------------------------------
# - C: TransposeD

class TransposeD(mn.transforms.Transform):
    """A MONAI transform to transpose the dimensions of an input array.
    """
    def __init__(self, keys:list[str], indices:tuple) -> None:
        """Initialize the transform.
        
        Args: 
            keys (list(str)): the input MONAI keys to the transformation class.
            indices (tuple): the indices to transpose.
        """
        super().__init__()
        self.keys=keys
        self.transposer=mn.transforms.Transpose(indices)

    def __call__(self, data: dict) -> dict:
        """Body of the transformation.
        
        Args:
            data (Dict): a dictionary of data to be transformed with MONAI.
        
        Returns:
            data_copy (Dict): the transformed data.
        """
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img=data[key].copy()
                data_copy[key]=self.transposer(img)
                data_copy[key]=img
        return data_copy

#---------------------------------------
# - C: ConvertToPIL
    
class ConvertToPIL(mn.transforms.Transform):
    """ A MONAI transform to convert a PyTorch tensor or a NumPy array 
    to a PIL image.
    """
    def __init__(self, mode:str="RGB") -> None:
        """Initialize the transform.

        Args:
            mode (str, optional): The mode of conversion for PIL. 
                Defaults to "RGB".
        """
        super().__init__()
        self.mode=mode.upper()

    def __call__(self, data: Union[torch.Tensor, 
                                   np.ndarray]) -> PILImage.Image:
        """_summary_

        Args:
            data (Union[torch.Tensor, np.ndarray]): _description_

        Returns:
            img (PILImage.Image): The converted image.
        """
        img=data.copy()
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if self.mode=="RGB":
            if len(img.shape)==2:
                img = np.expand_dims(img,axis=-1)
                img = np.concatenate([img,img,img], axis=2)
            elif len(img.shape)==3:
                if img.shape[0]==1 or img.shape[0]==3:
                    img = img.transpose(1,2,0)
                if img.shape[-1]==1:
                    img = np.concatenate([img,img,img], axis=2)
        elif self.mode=="L":
            if len(img.shape)==2:
                img = np.expand_dims(img,axis=-1)
            elif len(img.shape)==3:
                if img.shape[-1]==3:
                    img = np.mean(img, axis=-1)
                elif img.shape[0]==1:
                    img = img.transpose(1,2,0)
                elif img.shape[0]==3:
                    img = np.mean(img, axis=0)
                    img = img.transpose(1,2,0)
        
        img = img * 255
        img = img.astype('uint8')
        img = PILImage.fromarray(img, self.mode)
        return img

#---------------------------------------
# - C: RandAugD

class RandAugD(mn.transforms.RandomizableTransform):
    """ A MONAI transform to apply random augmentations to an image.
    """
    def __init__(self, 
                 keys:list[str], 
                 pil_conversion_mode:str = "RGB", 
                 m:int=5, n:int=2, mstd:float=0.5, 
                 convert_to_numpy:bool=True) -> None:
        """Initialize the transform.

        Args:
            keys (list(str)): the input MONAI keys to the transformation class.
            pil_conversion_mode (str, optional): The mode for conversion. 
                Defaults to "RGB".
            m (int, optional): magnitude of augmentations. Defaults to 9.
            n (int, optional): number of augmentations. Defaults to 2.
            mstd (float, optional): the standard deviation of the magnitude 
                noise applied. Defaults to 0.5.
            convert_to_numpy (bool, optional): whether to convert the input
                data to NumPy array. Defaults to True.
        """
        super().__init__()
        self.keys=keys
        self.converter = ConvertToPIL(mode=pil_conversion_mode)
        self.convert_to_numpy = convert_to_numpy
        timm.data.auto_augment._RAND_TRANSFORMS  = [
            'AutoContrast',
            'Equalize',
            #'Invert',
            'Rotate',
            #'Posterize',
            #'Solarize',
            #'SolarizeAdd',
            #'Color',
            'Contrast',
            'Brightness',
            'Sharpness',
            'ShearX',
            'ShearY',
            'TranslateXRel',
            'TranslateYRel',
        ]
        self.augmentor = timm.data.auto_augment.rand_augment_transform(
            config_str=f"rand-n{n}-m{m}-mstd{mstd}", 
            hparams={'img_mean': 0})
        
    def __call__(self, data: dict) -> dict:
        """Body of the transformation.
        
        Args:
            data (Dict): a dictionary of data to be transformed with MONAI.
        
        Returns:
            data_copy (Dict): the transformed data.
        """
        data_copy=copy.deepcopy(data)
        for key in data:
            if key in self.keys:
                img = data[key].copy()
                img = self.converter(img)
                img = self.augmentor(img)
                if self.convert_to_numpy:
                    img = np.array(img) 
                    img = np.mean(img, axis=-1)
                    img = np.expand_dims(img, axis=0)
                data_copy[key] = img
        return data_copy

#-------------------------------------------------------------------------------
# MONAI helper functions
#-------------------------------------------------------------------------------

#---------------------------------------
# - F: empty_monai_cache

def empty_monai_cache(cache_dir:str) -> None:
    """Empty the MONAI cache directory.

    Args:
        cache_dir (str): The MONAI cache directory.
    """
    if os.path.exists(cache_dir+"/train"):
        shutil.rmtree(cache_dir+"/train")
        print("MOANI's train cache directory removed successfully!")

    if os.path.exists(cache_dir+"/val"):
        shutil.rmtree(cache_dir+"/val")
        print("MOANI's validation cache directory removed successfully!")
        
    if os.path.exists(cache_dir+"/test"):
        shutil.rmtree(cache_dir+"/test")
        print("MOANI's test cache directory removed successfully!")