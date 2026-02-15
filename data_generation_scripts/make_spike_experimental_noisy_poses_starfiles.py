import numpy as np
import pandas as pd
import starfile
import copy

# Modifying pose data in star files
def new_pose_data_star(star_file, root_dir):
    print(root_dir + star_file)
    file = open(root_dir + star_file, 'r')
    data_frame_star = starfile.read(root_dir+star_file)
    return data_frame_star 

def main():
    
    # Load up star file
    root_dir = "/mnt/home/levans/ceph/EMPIAR-12098/csparc_pruned_mixed/"
    star_file = "particles_fourier_downsampled.star"

    # Load up angles and shifts
    data_frame_star = new_pose_data_star(star_file, root_dir)
    angle_rot = np.array(data_frame_star['particles']['rlnAngleRot'])
    angle_tilt = np.array(data_frame_star['particles']['rlnAngleTilt'])
    angle_psi = np.array(data_frame_star['particles']['rlnAnglePsi'])
    all_angles = np.stack([angle_rot, angle_tilt, angle_psi])

    shifts_x = np.array(data_frame_star['particles']['rlnOriginXAngst'])
    shifts_y = np.array(data_frame_star['particles']['rlnOriginYAngst'])
    all_shifts = np.stack([shifts_x, shifts_y])

    # Add uniform noise to all rotation angles
    scales = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    for i in range(len(scales)):
        new_data_frame_star = copy.deepcopy(data_frame_star)
        fields = ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']
        scale = scales[i]
        for field in fields: 
            angles = data_frame_star['particles'][field].copy()
            noise = np.random.uniform(low=-scale, high=scale, size=angle_rot.shape)
            print(np.std(angles))
            if field == 'rlnAngleTilt': 
                noisy_angles = np.mod(angles + noise, 180)
            else: 
                noisy_angles = np.mod(angles + noise + 180, 360) - 180
            new_data_frame_star['particles'][field] = noisy_angles 

        print("done with angles") 
        starfile.write(new_data_frame_star, root_dir + f"particles_fourier_downsampled_noisy_poses_magn_{scale}.star")

    # Now doing shifts
    scales = [3, 3.25, 3.5, 3.75, 4]
    for i in range(len(scales)):
        new_data_frame_star = copy.deepcopy(data_frame_star)
        fields = ['rlnOriginXAngst', 'rlnOriginYAngst']
        scale = scales[i]
        std_dev = scale
        #std_dev = scale*std_images
        for field in fields: 
            noise = np.random.uniform(low=-scale, high=scale, size=angle_rot.shape)
            shifts = data_frame_star['particles'][field].copy()
            noisy_shifts = shifts + noise 
            new_data_frame_star['particles'][field] = noisy_shifts

        print("done with shifts") 
        starfile.write(new_data_frame_star, root_dir + f"particles_fourier_downsampled_noisy_shifts_level_{i}.star")

if __name__ == "__main__":
    main()