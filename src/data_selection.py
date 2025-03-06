import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import logging
import scipy.io as sio
from pathlib import Path


class Sampling:
    def __init__(self, data_path="HRRT_NX", *kwargs):
        self.data_path = data_path

    def enhance_constrast(self, img):
        """
        Enhance the contrast of the input image using histogram equalization.

        Args:
            img: numpy array representing the input image.

        Returns:
            numpy array representing the enhanced image.
        """
        return cv2.equalizeHist(img.astype(np.uint8))

    def display_montage(
        self,
        img,
        grid_size=(8, 16),
        display_range=(0, 100000),
        fig_name="montage.png",
        non_voxel=False,
    ):
        # slices = extract_slices(img, method="scale_max")
        """
        Display a montage of 2D slices from the input 3D image.

        Parameters
        ----------
        img : numpy array
            Input 3D image.
        grid_size : tuple, optional
            Size of the grid for the montage, by default (8, 16).
        display_range : tuple, optional
            Range for displaying the image, by default (0, 100000).
        fig_name : str, optional
            Name of the output figure file, by default "montage.png".

        Returns
        -------
        None
        """
        slices = np.linspace(
            0, img.shape[2] - 1, grid_size[0] * grid_size[1], dtype=int
        )  # Select evenly spaced slices
        print(slices)
        plt.figure(figsize=(10, 10))
        for idx, slice in enumerate(slices):
            plt.subplot(grid_size[0], grid_size[1], idx + 1)
            slice_2d = img[:, :, slice]  # Extract slice
            if non_voxel:
                slice_2d = img[slice]

            plt.imshow(
                slice_2d, cmap="gray", vmin=display_range[0], vmax=display_range[1]
            )
            plt.axis("off")
            plt.title(f"Slice {slice}", fontsize=8)

        plt.savefig(fig_name, dpi=300, bbox_inches="tight")  # Save as PNG
        plt.close()  # Close the figure to free memory

    def zscore(self, data, threshold=3):
        """
        Identify indices of data points that lie within a specified z-score threshold.

        Args:
            data (np.ndarray): The input data array.
            threshold (float, optional): The z-score threshold for identifying core points.
                Points with an absolute z-score less than or equal to this value are considered
                core points. Default is 3.

        Returns:
            np.ndarray: Indices of the data points that are within the specified z-score threshold.
        """
        mean = np.mean(data)
        std = np.std(data)
        z_scores = (data - mean) / std
        condition = np.abs(z_scores) <= threshold
        core_points = data[condition]
        indices = np.where(condition)[0]

        return indices

    def iqr(self, data):
        """
        Identify indices of data points that lie within the interquartile range (IQR).

        Parameters
        ----------
        data : np.ndarray
            The input data array.

        Returns
        -------
        np.ndarray
            Indices of the data points that are within the IQR.
        """
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        condition = data > lower_bound
        core_points = data[condition]
        indices = np.where(condition)[0]
        return indices

    def extract_slices(self, img, method="z-score", scale_max=0.10):
        """
        Extract slices from a 3D image based on the specified method.

        Parameters
        ----------
        img : numpy array
            Input 3D image.
        method : str, optional
            Method for extracting slices. Options are 'z-score', 'iqr', 'scale_max', and 'gradient_filtering'.
            Default is 'z-score'.
        scale_max : float, optional
            Percentage of the maximum value for scaling within a certain range. Default is 0.25.

        Returns
        -------
        list
            Indices of the extracted slices.
        """
        slice_indices = np.array([idx for idx in range(img.shape[2])])
        slice_values = np.array([np.std(img[:, :, slice]) for slice in slice_indices])
        if method == "z-score":
            indices = self.zscore(slice_values)
        elif method == "iqr":
            indices = self.iqr(slice_values)
        elif method == "scale_max":
            """
            Scale within a percentage of max value. Default: scale_max = 0.20
            """
            scale_max = scale_max
            slice_mean = np.mean(img, axis=(0, 1))
            slice_max = np.max(slice_mean)
            threshold = scale_max * slice_max
            indices = np.where(slice_mean > threshold)[0]
            slice_mean_filtered = slice_mean[slice_mean > threshold]

        elif method == "gradient_filtering":
            """
            TODO: Experimental. Implement this later for optimzation of data collection.
            Filter using gradient to remove near flat or zero change regions in the image.
            """
            pass

        return indices

    def examine(self, img):
        """
        Analyze and print statistics of a 3D image.

        This function computes and prints various statistics of the input image,
        including the shape of the image, sums and means of the middle slice, and
        maximum sum slice. It also calculates and prints the standard deviations
        of the sums and means of all slices.

        Parameters
        ----------
        img : numpy array
            Input 3D image.

        Returns
        -------
        None
        """
        print("Shape of the image: ", img.shape)

        mid_slice = img[:, :, img.shape[2] // 2]
        print("Sum of middle slice: ", np.sum(mid_slice))
        print("Mean of the middle slice: ", np.mean(mid_slice))

        slice_sums = np.sum(img, axis=(0, 1))  # Sum over height & width, keeping depth
        slice_means = np.mean(
            img, axis=(0, 1)
        )  # Mean over height & width, keeping depth

        # Find the index of the slice with the maximum sum
        max_sum_index = np.argmax(slice_sums)

        # Extract the slice with the max sum
        max_sum_slice = img[:, :, max_sum_index]
        print("Sum of Max Slice: ", np.sum(max_sum_slice))
        print("Mean of Max Slice: ", np.mean(max_sum_slice))

        sum_std_dev = np.std(slice_sums)
        print("Standard Dev of sums: ", sum_std_dev)

        mean_std_dev = np.std(slice_means)
        print("Standard Dev of means: ", mean_std_dev)

    def plot_distribution(self):
        """
        Plot the mean distribution of the top-view slices in different frames.

        This function loads three nifti files from the data_path directory, computes
        the mean of each slice along the height and width axes, and plots the
        distribution of the means using matplotlib. The x-axis represents the
        slice index, and the y-axis represents the mean value of the slice.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        file_path_1 = self.data_path / "resliced_frame_089.hdr"
        file_path_2 = self.data_path / "resliced_frame_059.hdr"
        file_path_3 = self.data_path / "resliced_frame_001.hdr"

        img_1 = nib.load(file_path_1)
        img_2 = nib.load(file_path_2)
        img_3 = nib.load(file_path_3)

        img_1 = img_1.get_fdata()
        img_2 = img_2.get_fdata()
        img_3 = img_3.get_fdata()

        slice_means_1 = np.mean(
            img_1, axis=(0, 1)
        )  # Mean over height & width, keeping depth
        slice_means_2 = np.mean(img_2, axis=(0, 1))
        slice_means_3 = np.mean(img_3, axis=(0, 1))

        slice_indices = np.array([idx for idx in range(img_1.shape[2])])
        plt.figure(figsize=(8, 5))
        plt.scatter(
            slice_indices, slice_means_1, color="red", alpha=0.5, s=8, label="069"
        )
        plt.scatter(
            slice_indices, slice_means_2, color="blue", alpha=0.5, s=8, label="059"
        )
        plt.scatter(
            slice_indices, slice_means_3, color="green", alpha=0.5, s=8, label="001"
        )

        # Labels and title
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        plt.title("Mean-distribution of top-view slices in different frames")
        plt.legend()
        plt.savefig("means_distribution.png", dpi=300, bbox_inches="tight")
        # plt.show()

    def check_image(self, img_path):
        """
        Checks the image at the given path by loading it and printing it out.

        Args:
            img_path (str): The .hdr path to the image to be checked. 
        """
        img = nib.load(img_path)
        print(img)

    def pad_to_square(self, image): 
        """
        Pads an image to a squared shape by adding zeros to the top/bottom and left/right of the image.
        
        Parameters
        ----------
        image : numpy array
            The image to be padded.
        
        Returns
        -------
        numpy array
            The padded image in a squared shape.
        """
        H, W = image.shape[:2]  # Get original height and width
        new_size = max(H, W)  # Determine the target size

        # Calculate padding for height and width
        pad_top = (new_size - H) // 2
        pad_bottom = new_size - H - pad_top
        pad_left = (new_size - W) // 2
        pad_right = new_size - W - pad_left

        # Apply padding
        padded_image = np.pad(image, 
                            ((pad_top, pad_bottom), (pad_left, pad_right)), 
                            mode='constant', constant_values=0)

        return padded_image
    
    def pad_to_size(self, image, size=512): 
        """
        Pads an image to a specified size by adding zeros to the top/bottom and left/right of the image.
        
        Parameters
        ----------
        image : numpy array
            The image to be padded.
        size : int, optional
            The size to which the image should be padded, by default 512
            
        Returns
        -------
        numpy array
            The padded image.
        """
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_h = max(size - h, 0) // 2
        pad_w = max(size - w, 0) // 2
        
        # Add extra pixel to bottom/right if odd number of padding pixels
        pad_h_extra = max(size - h - 2*pad_h, 0)
        pad_w_extra = max(size - w - 2*pad_w, 0)
        
        # Pad the image
        padded_image = np.pad(image,
                            ((pad_h, pad_h + pad_h_extra),
                            (pad_w, pad_w + pad_w_extra)),
                            mode='constant', constant_values=0)
        
        return padded_image
    
    def generate_dataset(self, option='npy'):
        """
        TODO: Add logging to this function
        Generates a dataset from a list of 3D images.

        This function loads a list of 3D images, extracts slices from each image
        using the extract_slices method, and stores the extracted slices in a
        single numpy array. The array is then saved to a file using numpy's save
        function.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        # files = [file_path for file_path in self.data_path.rglob("*.hdr")][0:5]
        files = [file_path for file_path in self.data_path.rglob("*.hdr")]
        frames_data = []  # Initialize an empty list to store core data arrays

        for file in files:
            img_data = nib.load(file).get_fdata()
            indices = self.extract_slices(img_data, method="scale_max")
            for idx in indices:
                core_data = img_data[:, :, idx]
                core_data = core_data / np.max(core_data)
                H, W = core_data.shape

                # Padding if the image is not squared
                if H != W: core_data = self.pad_to_square(core_data)

                print("Data Shape: ", core_data.shape)
                frames_data.append(core_data)
        frames_data = np.array(frames_data)
        frames_data.shape
        frames_data = frames_data.transpose(1, 2, 0)  # Reorder dimensions

        if option == 'npy':
            print("Final Data Shape: ", frames_data.shape)
            np.save(Path("../data") / "data.npy", frames_data)
        elif option == 'mat': 
            print("Final Data Shape: ", frames_data.shape)
            sio.savemat(Path("../data") / "data.mat", {"images": frames_data})


class TestImages(Sampling): 
    def __init__(self, test_path, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.test_path = test_path
        
    def get_test_images(self, dest_path): 
        files = [file_path for file_path in Path(self.test_path).rglob("*.hdr")]
        count = 0
        for file in files:
            img_data = nib.load(file).get_fdata()
            indices = self.extract_slices(img_data, method="scale_max")
            for idx in indices:
                core_data = img_data[:, :, idx]
                H, W = core_data.shape
                # Padding if the image is not squared
                if H != W: core_data = self.pad_to_size(core_data)
                output_path = Path(dest_path) / f"slice_{count}.png"
                plt.imsave(output_path, core_data, cmap="gray", vmin=0, vmax=5e4)
                count += 1
    

class Conditional_Sampling(Sampling): 
    def __init__(self, path_x, path_x_prior, output_path, *args, **kwargs): 
        super().__init__(*args, **kwargs)
        self.path_x = Path(path_x)
        self.path_x_prior = Path(path_x_prior)
        self.output_path = Path(output_path)
    
    def next_multiple_of_8(n):
        return ((n // 8) + 1) * 8
    
    #TODO: Add code to generate dual-dataset (NX (original data) + HRRT (prior data))
    def generate_conditional_dataset(self, option='npy'): 
        x_files = [file_path for file_path in self.path_x.rglob("*.hdr")]
        x_prior_files = [file_path for file_path in self.path_x_prior.rglob("*.hdr")]
        x_frames_data = []  # Initialize an empty list to store core data arrays
        x_prior_frames_data = []
        for x_file, x_prior_file in zip(x_files, x_prior_files):
            x_img_data = nib.load(x_file).get_fdata()
            x_prior_img_data = nib.load(x_prior_file).get_fdata()
            indices = self.extract_slices(x_img_data, method="scale_max")
            
            #NOTE: Guarantee that the two images are of the same size
            try:
                if x_img_data.shape != x_prior_img_data.shape:
                    raise ValueError(f"Shape mismatch: x_prior.shape = {x_prior_img_data.shape} != x.shape = {x_img_data.shape}")
                else:
                    print("Shapes match!")
            except Exception as e:
                print("Error during shape check:", e)

            for idx in indices:
                x_core_data = x_img_data[:, :, idx]
                x_core_data = x_core_data / np.max(x_core_data)

                x_prior_core_data = x_prior_img_data[:, :, idx]
                x_prior_core_data = x_prior_core_data / np.max(x_prior_core_data)
                
                H, W = x_core_data.shape

                # Padding if the image is not squared
                #NOTE: Assume that the two images are of the same size
                if H != W: 
                    x_core_data = self.pad_to_size(x_core_data, size=self.next_multiple_of_8(H)) 
                    x_prior_core_data = self.pad_to_size(x_prior_core_data, size=self.next_multiple_of_8(H))

                print("Data Shape: ", x_core_data.shape)
                x_frames_data.append(x_core_data)
                x_prior_frames_data.append(x_prior_core_data)

        x_frames_data = np.array(x_frames_data)
        x_frames_data = x_frames_data.transpose(1, 2, 0)  # Reorder dimensions
        x_prior_frames_data = np.array(x_prior_frames_data)
        x_prior_frames_data = x_prior_frames_data.transpose(1, 2, 0)  # Reorder dimensions

        print("Final X Data Shape: ", x_frames_data.shape)
        print("Final X Prior Data Shape: ", x_prior_frames_data.shape)
        if option == 'npy':
            np.save(self.output_path / "data.npy", x_frames_data)
            np.save(self.output_path / "data_prior.npy", x_prior_frames_data)
        elif option == 'mat': 
            sio.savemat(self.output_path / "data.mat", {"images": x_frames_data})
            sio.savemat(self.output_path / "data_prior.mat", {"images": x_prior_frames_data})

def main():
    parser = argparse.ArgumentParser(
        description="Extract slices from a 3D image based on different methods."
    )

    parser.add_argument("--data_path", "-d", type=str, required=True)
    args = parser.parse_args()

    folder_path = Path(args.data_path) / "NX_FDG" / "20231027"

    sampling = Sampling(data_path=folder_path)
    # sampling.plot_distribution()
    sampling.generate_dataset(option='mat')
    sampling.generate_dataset(option='npy')

    # data = np.load(r"C:\Users\luongcn\pet_ddpm\data\data.npy")
    # sampling.display_montage(
    #     data[500:1000, :, :],
    #     grid_size=(8, 16),
    #     display_range=(0, 1e4),
    #     fig_name="montage.png",
    #     non_voxel=True,
    # )


if __name__ == "__main__":
    main()
