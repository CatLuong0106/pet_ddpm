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

if __name__ == "__main__":
    check_mat_shape('/home/luongcn/pet_ddpm/data/data.mat')
    # check_hdr_image('/home/luongcn/pet_ddpm/raw_data/HRRT_NX_pair/HRRT_DH485/resliced_frame_000.hdr')