import numpy as np

def read_8bit_img_from_raw_data(filename, height, width):
    # Returns a numpy 2D array of 8bit integers representing image pixels
    # Open specified file
    with open(filename, "rb") as f:
        # Convert bytes read from file into a numpy array
        img = np.frombuffer(f.read(), dtype=np.ubyte)
        # This check is for images with some header, only raw bytes will be read, extra bytes will be sliced out
        if img.size > height * width:
            img = img[img.size-(height*width):]
        return img.reshape((height, width)).astype(np.int32)

def write_8bit_img_to_raw_data(img, filename):
    # Writes the argument numpy 2D array into a file by converting values into single bytes
    # Open Specified file
    with open(filename, "wb") as f:
        # Write bytes into file
        f.write(img.astype(np.ubyte).tobytes())
        print("Image was written to {0}".format(filename))

def normalize(img, ceil=255):
    # Returns a normalized version of the argument numpy 2D array
    # Checks if the image is already normalized
    if img.min() < 0 or img.max() > ceil:
        # Copy the input image
        normalized_img = img.copy()
        # Subtract the minimum of the input image
        normalized_img = normalized_img - img.min()
        # Multiplication of ceil to the ratios of the image pixels with respect to the image maximum
        normalized_img = (ceil * normalized_img / normalized_img.max())
        assert(normalized_img.max() <= ceil)
        return normalized_img.astype(np.int32)
    else:
        return img
    

            
    
    
    
def print_info_2d(array):
    print("Array info:\n\tSize:\t[{}, {}]\n\tMin:\t{}\n\tMax:\t{}\n\tSum:\t{}\n\tType:\t{}".format(array.shape[0], array.shape[1], array.min(), array.max(), array.sum(), array.dtype))
    
def print_info_1d(array):
    print("Array info:\n\tSize:\t{}\n\tMin:\t{}\n\tMax:\t{}\n\tSum:\t{}\n\tType:\t{}".format(len(array), array.min(), array.max(), array.sum(), array.dtype))