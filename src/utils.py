import scipy.io
from data_selection import Sampling

def check_mat_shape(mat_path):
    """
    Checks the shape of the data in the given .mat file.

    Args:
        mat_path (str): The path to the .mat file to be checked.

    Returns:
        None
    """
    mat_file = scipy.io.loadmat(mat_path)
    data = mat_file['images']
    print(data.shape)

def check_hdr_image(hdr_path):
    """
    Checks the image at the given path by loading it and printing it out.

    Args:
        hdr_path (str): The .hdr path to the image to be checked. 
    """
    obj = Sampling()
    obj.check_image(hdr_path)

def get_pad_value(imsize, patch_size): 
    """
    Calculates the padding size needed to ensure that the entire image is covered by patches.

    Args:
        imsize (int): The size of the image.
        patch_size (int): The size of each patch.

    Returns:
        int: The calculated padding value.
    """

    k = imsize // patch_size
    pad = (k + 1)*patch_size - imsize
    return pad

if __name__ == "__main__":
    check_mat_shape('/home/luongcn/pet_ddpm/data/train_conditional/data.mat')
    print("Pad Size is: ", get_pad_value(408, 51))
    # check_hdr_image('/home/luongcn/pet_ddpm/raw_data/HRRT_NX_pair/HRRT_DH485/resliced_frame_000.hdr')