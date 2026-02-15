import mrcfile
import numpy as np
import os, sys

def add_gaussian_noise(images, scale=0.5):
    """Adds random Gaussian noise to images."""

    # add std dev noise, relative to std of whole image stack
    std_images = np.std(images)
    std_dev = scale * std_images
    noise = np.random.normal(loc=0, scale=std_dev, size=images.shape)
    noisy_images = images + noise
    return noisy_images

def process_mrc_files(input_folder, input_files, output_folder, level, noise_levels):
    """Processes .mrc files: add noise, and save to output folder."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Find all .mrc files in the input folder
    for file_name in input_files:
        input_path = os.path.join(input_folder, file_name+".mrc")
        output_path = os.path.join(output_folder, f"noise_scale_{noise_levels[level]}_{file_name}.mrc")

        with mrcfile.open(input_path, permissive=True) as mrc:
            print(f"Processing {file_name}")
            images = mrc.data

            if level == "none":
                noisy_images = images
            else:
                level = int(level)   
                noise_scale = np.sqrt(noise_levels[level]) 
                noisy_images = add_gaussian_noise(images, noise_scale)
        #Save to a new .mrc file
        with mrcfile.new(output_path, overwrite=True) as mrc_out:
            mrc_out.set_data(noisy_images.astype(np.float32))
            print(f"Saved noisy images to {output_path}")

# Parameters
#noise_levels = np.logspace(-1, 1, 6)
#noise_levels = [16, 25, 100]
noise_levels = [100]
input_folder = 'csparc_pruned_mixed'  # Folder containing the original .mrc files
input_files = ['particles_fourier_downsampled']  # Folder containing the original .mrc files
output_folder = sys.argv[1]            # Folder to save the processed .mrc files
level = int(sys.argv[2])                    

# Run the processing
process_mrc_files(input_folder, input_files, output_folder, level, noise_levels)
