import imageio
import nibabel as nib
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def enhance_constrat(img): 
    return cv2.equalizeHist(img.astype(np.uint8))

def display_montage(img, grid_size=(8, 16), display_range=(0, 100000)): 
    slices = np.linspace(0, img.shape[2] - 1, grid_size[0] * grid_size[1], dtype=int)  # Select evenly spaced slices
    plt.figure(figsize=(10,10))
    for idx,slice in  enumerate(slices):
      plt.subplot(grid_size[0], grid_size[1], idx+1)
      slice_2d = img[:, :, slice]  # Extract slice
      # slice_2d = np.clip(slice_2d, display_range[0], display_range[1])  # Apply display range
      plt.imshow(slice_2d, cmap="gray",vmin=display_range[0],vmax=display_range[1])
      plt.axis("off")
      plt.title(f"Slice {slice}",fontsize=8)

    plt.savefig("slices_nx_visualization.png", dpi=300, bbox_inches="tight")  # Save as PNG
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
    print(slice_values)
    if method == 'z-score': 
        print(find_outliers_zscore(slice_values))
    elif method == 'iqr': 
        print(find_outliers_iqr(slice_values))  # Output: [100 200]

# Call the function
def main(): 
    data_path = Path("HRRT_NX")
    file_path = data_path / "HRRT_FDG" / "20231023_DH485" /"resliced_frame_069.hdr"
    img = nib.load(file_path) 
    data = img.get_fdata()
    display_montage(data, grid_size=(8, 16), display_range=(0, 1e5))
    extract_slices(data, method='z-score')

if __name__ == "__main__": 
    main()