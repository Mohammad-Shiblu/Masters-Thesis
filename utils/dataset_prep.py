import numpy as np
import os 
import pydicom
import json

def prep_dataset(config): 
    save_dir = os.path.join(config['save_dir'], config['slice_thickness']+" "+ config['reconstruction_kernel'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory path create: {save_dir}")

    data_dir = os.path.join(config['data_dir'], config['slice_thickness']+" "+ config['reconstruction_kernel'])

    input_dir = os.path.join(data_dir, "QD_"+config['slice_thickness'], "quarter_"+config['slice_thickness'])
    target_dir = os.path.join(data_dir, "FD_"+config['slice_thickness'], "full_"+config['slice_thickness'])

    patients_list = sorted(os.listdir(input_dir))
    
    for patient in patients_list:
        patient_input_path = os.path.join(input_dir, patient, "quarter_"+config['slice_thickness'])
        patient_target_path = os.path.join(target_dir, patient, "full_"+config['slice_thickness'])

        for path in [patient_input_path, patient_target_path]:
            if not os.path.exists(path):
                print(f"Path does not exist: {path}")
            
            all_slices = HU_converted(load_scan(path)) # return a 3D array consisting all the slices
            for slice_num in range(len(all_slices)):
                dose = "input_low_dose" if "QD" in path else 'target_full_dose'
                slice = normalize(all_slices[slice_num], config)
                slice_name = f"{patient}_{dose}_{slice_num:03d}.npy"
                np.save(os.path.join(save_dir, slice_name), slice)

        print(f"{patient} data has been processed successfully")

    print("Data processing and dataset preparation are completed")       


# datatset preparation with gaussian noise and block distorted part
def prep_dataset_gaussian_noise(config):
    save_dir = os.path.join(config['save_dir'], config['slice_thickness']+" "+ config['reconstruction_kernel'])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Directory path create: {save_dir}")

    data_dir = os.path.join(config['data_dir'], config['slice_thickness']+" "+ config['reconstruction_kernel'])

    
    final_data_dir = os.path.join(data_dir, "FD_"+config['slice_thickness'], "full_"+config['slice_thickness'])
    patient_list = sorted(os.listdir(final_data_dir))
    # print(patient_list)

    for patient in patient_list:
        patient_target_path = os.path.join(final_data_dir, patient, "full_"+config['slice_thickness']) # data file inside individual patient
        
        if not os.path.exists(patient_target_path):
            print(f"Patient path does not exist: {patient_target_path}")
            continue
        all_slices = HU_converted(load_scan(patient_target_path))
        
        for slice_num in range(len(all_slices)):
            slice = normalize(all_slices[slice_num], config)
            target_name = f"{patient}_target_{slice_num:03d}.npy"
            np.save(os.path.join(save_dir, target_name), slice)
            # prep input image
            noise_image = add_noise(slice)
            patched_image = add_patches(noise_image)
            input_name = f"{patient}_input_{slice_num:03d}.npy"
            np.save(os.path.join(save_dir, input_name), patched_image)

        print(f"{patient} data has been processed successfully")

    print("Data processing and dataset preparation are completed") 

    
# function for adding black and white patches into image
def add_patches(image, num_patches = 5, size_range = (30, 40)):
    patched_image = image.copy()
    height, width = patched_image.shape
    num = np.random.randint(1, num_patches)
    for _ in range(num):
        patch_height = np.random.randint(size_range[0], size_range[1])
        patch_width = np.random.randint(size_range[0], size_range[1])

        top_left_y = np.random.randint(0, height -patch_height)
        top_left_x = np.random.randint(0, width - patch_width)

        if np.random.rand() > 0.5:
            patch_color = 0
        else:
            patch_color = 1
        
        patched_image[top_left_y: top_left_y + patch_height, top_left_x: top_left_x + patch_width] = patch_color

    return patched_image



# function for adding noise into image
def add_noise(image, mean=0.0, std=0.4, peak_counts=100):
    lam = image*float(peak_counts)
    poisson_noisy = np.random.poisson(lam).astype(np.float32) / float(peak_counts)
    # --gaussian noise----
    gauss = np.random.normal(loc=mean, scale=std, size=image.shape)
    noisy_image = poisson_noisy + gauss
    noisy_image = np.clip(noisy_image, 0.0, 1.0)
    return noisy_image

# sort the slices based on the ImagePositionPatient[2] attribute or z axis position
def load_scan(path):
    slices = [pydicom.dcmread(os.path.join(path, s)) for s in os.listdir(path)]
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation) 
    
    # for s in slices:
    #     # s.SliceThickness = slice_thickness
    #     pass
    return slices

# convert the pixel values to Hounsfield Units (HU): pixel_value = slope * pixel_value + intercept
def HU_converted(slices):
    image = np.stack([s.pixel_array for s in slices])
    image = image.astype(np.float32)
    image[image == -2000] = 0  # set background to 0
    for slice_num in range(len(slices)):
        intercept = slices[slice_num].RescaleIntercept
        slope = slices[slice_num].RescaleSlope
        if slope == 0:
            raise ValueError(f"Invalid slope = 0 for slice {slice_num}")
        elif slope != 1:
            image[slice_num]= slope * image[slice_num].astype(np.float32)
            image[slice_num] = image[slice_num].astype(np.float32)
        
        image[slice_num] += np.float32(intercept)
    
    return np.array(image, dtype=np.float32)  
        

# CT image normalization
def normalize(image, config):
    norm_min = config['norm_min']
    norm_max = config['norm_max']
    
    norm_image = (image - norm_min) / (norm_max - norm_min)
    return norm_image

        


if __name__ == '__main__':
    with open('config/data_prep.json', 'r') as f:
        config = json.load(f)
    
    # prep_dataset(config)
    prep_dataset_gaussian_noise(config)




    

