import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Load the Analyze format image
file_path = r"C:\Users\luongcn\PaDIS\HRRT_NX\HRRT_FDG\20231023_DH485\resliced_frame_000.hdr"
img = nib.load(file_path)
data = img.get_fdata()

# Display a slice (middle slice along the Z-axis)
plt.imshow(data[:, :, data.shape[2] // 2], cmap="hot")
plt.colorbar()
plt.title("PET/MRI/CT Scan Slice")
plt.show()