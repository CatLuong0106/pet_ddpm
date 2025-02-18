import imageio
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def enhance_constrat(img): 
    return cv2.equalizeHist(img.astype(np.uint8))

def display_montage(img, grid_size=(8, 16), display_range=(0, 100000)): 
    # slices = extract_slices(img, method="scale_max")
    # six_slices = slices[::6]
    # print(six_slices)
    slices = np.linspace(0, img.shape[2] - 1, grid_size[0] * grid_size[1], dtype=int)  # Select evenly spaced slices
    print(slices)
    plt.figure(figsize=(10,10))
    for idx,slice in enumerate(slices):
      plt.subplot(grid_size[0], grid_size[1], idx+1)
      slice_2d = img[:, :, slice]  # Extract slice
    #   print(slice_2d.shape)
    #   slice_2d = np.clip(slice_2d, display_range[0], display_range[1])  # Apply display range
      plt.imshow(slice_2d, cmap="gray",vmin=display_range[0],vmax=display_range[1])
      plt.axis("off")
      plt.title(f"Slice {slice}",fontsize=8)

    plt.savefig("phno_visualization.png", dpi=300, bbox_inches="tight")  # Save as PNG
    plt.close()  # Close the figure to free memory

def find_outliers_zscore(data, threshold=3):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = (data - mean) / std
    outliers = data[np.abs(z_scores) > threshold]
    return outliers

def find_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    outliers = data[(data <= lower_bound)]
    return outliers

def extract_slices(img, method='z-score'): 
    slice_indices = np.array([idx for idx in range(img.shape[2])])
    slice_values = np.array([np.std(img[:, :, slice]) for slice in slice_indices])
    if method == 'z-score': 
        print(find_outliers_zscore(slice_values))
    elif method == 'iqr': 
        print(find_outliers_iqr(slice_values))  # Output: [100 200]
    elif method == "scale_max": 
        """
        Scale within a percentage of max value. Default: scale_max = 0.20
        """
        scale_max = 0.25
        slice_mean = np.mean(img, axis=(0, 1))
        slice_max = np.max(slice_mean)
        threshold = scale_max * slice_max
        indices = np.where(slice_mean > threshold)
        slice_mean_filtered = slice_mean[slice_mean > threshold]
        # print("Number of slices remained: ", len(slice_mean_filtered))
        # print("The indices remained: ", indices)
        return indices[0]
        
        
    elif method == "gradient_filtering": 
        """
        Filter using gradient to remove near flat or zero change regions in the image. 
        """
        dy = np.gradient(y)
        filtered_indices = np.abs(dy) > 10  # Remove near-flat regions
        pass



def examine(img): 
    print("Shape of the image: ", img.shape)
    slice_indices = np.array([idx for idx in range(img.shape[2])])
    mid_slice = img[:, :, img.shape[2] // 2]
    print("Sum of middle slice: ", np.sum(mid_slice)) 
    print("Mean of the middle slice: ", np.mean(mid_slice))
    
    slice_sums = np.sum(img, axis=(0, 1))  # Sum over height & width, keeping depth
    slice_means = np.mean(img, axis=(0, 1))  # Mean over height & width, keeping depth
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

def plot_distribution(img_1, img_2, img_3):
    slice_means_1 = np.mean(img_1, axis=(0, 1))  # Mean over height & width, keeping depth
    slice_means_2 = np.mean(img_2, axis=(0, 1))
    slice_means_3 = np.mean(img_3, axis=(0, 1))
    
    slice_indices = np.array([idx for idx in range(img_1.shape[2])])
    plt.figure(figsize=(8, 5))
    plt.scatter(slice_indices, slice_means_1,color='red', alpha=0.5, s=8, label="069")
    plt.scatter(slice_indices, slice_means_2,color='blue', alpha=0.5, s=8, label="059")
    plt.scatter(slice_indices, slice_means_3,color='green', alpha=0.5, s=8, label="001")

    # Labels and title
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Mean-distribution of top-view slices in different frames")
    plt.legend()
    plt.savefig("means_distribution.png", dpi=300, bbox_inches="tight")
    # plt.show()
# Call the function

def check_hdr(img_path):
    img = nib.load(img_path)
    print(img) 

def main(): 
    
    data_path = Path("data")
    file_path_1 = data_path / "resliced_frame_069.hdr"
    file_path_2 = data_path / "resliced_frame_059.hdr"
    file_path_3 = data_path / "resliced_frame_001.hdr"
    file_path = data_path / "mc_frame_000.hdr"
    data = nib.load(file_path).get_fdata()
    img_1 = nib.load(file_path_1) 
    img_2 = nib.load(file_path_2)
    img_3 = nib.load(file_path_3)

    
    data_1 = img_1.get_fdata()
    data_2 = img_2.get_fdata()
    data_3 = img_3.get_fdata()
    
    display_montage(data, grid_size=(8, 16), display_range=(0, 10))
    # extract_slices(data_1, method='scale_max')
    # examine(data)
    # plot_distribution(data_1, data_2, data_3)
    print(nib.load(file_path_1).get_fdata().shape)
    check_hdr(file_path)
    
if __name__ == "__main__": 
    main()