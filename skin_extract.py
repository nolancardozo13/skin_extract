import numpy as np
from PIL import Image
from skimage import filters, morphology
from scipy import ndimage
import argparse

def extract_body(img_path, out_path, sigma, remove_pixels):
    '''
    This algorithm automatically extracts the skin pixels 
    from the background using several image processing 
    operations.
    input:
        img_path: The path to the input image (str)
        out_path: The path to store the segmented images (str)
        sigma: the standard deviation for the gaussian filter (float)
        remove_pixels: The maximum size of the pixel islands 
                       to be removed after segmentation (int)

    Output:
        segmented image stored at out_path

    '''
    # Input the origin RGB image
    image = Image.open(img_path)    
    # The skin segmentation works best in the Cr channel of the 
    # YCbCr color space, so extract the Cr channel
    image_cr = image.convert('YCbCr').split()[2]
    # Smooth the image using Gaussian filter to remove noise resulting in 
    # better segmentation
    image_cr = filters.gaussian(np.array(image_cr), sigma = sigma)
    # Use OTSU thresholding for finding the optimal threshold
    val = filters.threshold_otsu(image_cr)
    # Apply threshold to derive the binary mask
    mask = image_cr > val
    # Remove small isolated pixel islands
    mask = morphology.remove_small_objects(mask,remove_pixels)
    # Fill in small holes 
    mask = ndimage.binary_fill_holes(mask)
    # Add a third channel to the mask
    y = np.expand_dims(mask,axis = 2)
    new_mask = np.concatenate((y,y,y),axis = 2)
    # Get the original RGB image with the background in the mask
    # set to 0
    image = Image.fromarray(image.convert('RGB') * new_mask)
    image.save(out_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type = str, help = "The path to the input image")
    parser.add_argument("--out_path", default = 'output.jpg', type = str, help = "The path to the save the output image")
    parser.add_argument("--sigma", default = 3, type = float, help= "The sigma for the gaussian smoothening before segmentaion")
    parser.add_argument("--remove_pixels", default = 500 , type = int, help= "The maximum size of pixels islands to be exlcuded after segmentation")
    args = parser.parse_args()
    extract_body(args.image_path, args.out_path, args.sigma, args.remove_pixels)