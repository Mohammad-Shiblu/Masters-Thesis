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
def add_patches(image, num_patches=3, size_range=(5, 10)):
    # Reduced patch size (5-10px vs 30-40px) and count (max 3 vs 5).
    # Large patches (30-40px) destroy anatomy over 12-15% of the image width,
    # making internal structure unrecoverable. Small patches simulate
    # minor detector artifacts without eliminating structural content.
    patched_image = image.copy()
    height, width = patched_image.shape
    num = np.random.randint(1, num_patches)
    for _ in range(num):
        patch_height = np.random.randint(size_range[0], size_range[1])
        patch_width = np.random.randint(size_range[0], size_range[1])

        top_left_y = np.random.randint(0, height - patch_height)
        top_left_x = np.random.randint(0, width - patch_width)

        patch_color = 0 if np.random.rand() > 0.5 else 1
        patched_image[top_left_y: top_left_y + patch_height, top_left_x: top_left_x + patch_width] = patch_color

    return patched_image



# function for adding noise into image
def add_noise(image, mean=0.0, std=0.05, peak_counts=10000):
    # Poisson noise: simulates photon counting noise at quarter-dose level.
    # peak_counts=10000 gives SNR~100 (realistic for low-dose CT).
    lam = image * float(peak_counts)
    poisson_noisy = np.random.poisson(lam).astype(np.float32) / float(peak_counts)
    # Gaussian noise: models electronic/readout noise.
    # std=0.05 on [0,1] is realistic (contrast differences in CT are ~0.05-0.15).
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

        


def add_ct_noise_standard(image, I0=1000, sigma_e=0.002):
    """
    Standard research-paper CT noise model: Poisson (quantum) + Gaussian (electronic).

    This follows the image-domain noise model used in the majority of LDCT denoising
    papers (RED-CNN Chen et al. 2017, WGAN-VGG Yang et al. 2018, FFDNet, etc.) for
    studies that synthesise low-dose CT without access to raw sinogram data.

    Model
    -----
        noisy = Poisson(image * I0) / I0  +  N(0, sigma_e²)

    Poisson term (I0 — incident photon count)
        Quantum / shot noise: the dominant noise source in CT.
        σ_Poisson(x) = sqrt(x / I0)  at normalised pixel value x.

        At water (x≈0.25, HU=0 with [-1024,3072] norm):
            I0=1000  → σ ≈ 0.016  ≈  65 HU   (simulated ~10% dose)  → ~35 dB
            I0=2500  → σ ≈ 0.010  ≈  41 HU   (simulated ~25% dose)  → ~40 dB
            I0=5000  → σ ≈ 0.007  ≈  29 HU   (simulated ~50% dose)  → ~43 dB
        The real Mayo QD dataset (1mm B30) has σ≈19 HU → baseline PSNR ~46 dB.
        Default I0=1000 gives a clearly challenging task (~35 dB) without being
        physically unrealistic (compare: old add_noise() Gaussian std=0.05 ≈ 205 HU).

    Gaussian term (sigma_e — electronic / readout noise)
        Real CT scanners: 2–5 HU ≈ 0.0005–0.001 (normalised).
        sigma_e=0.002 ≈ 8 HU: physically realistic, small relative to Poisson.
        The old add_noise() used std=0.05 (≈205 HU), which is unphysical.

    Why Poisson should dominate
        CT noise is fundamentally photon-counting noise.  A physically correct model
        has signal-dependent (Poisson) variance, not flat (Gaussian) variance.
        This matters because networks trained on pure Gaussian noise can fail to
        generalise to the signal-dependent noise patterns in real clinical CT.

    Note: the model uses `image * I0` as the expected photon count.  In strict
    CT physics, denser tissue (bright pixels) attenuates MORE photons, so transmitted
    counts would be lower for high-HU tissue.  Without raw sinogram data, this
    image-domain inversion is the standard simplification used across the literature.

    Args:
        image   : Normalised CT slice in [0, 1].
        I0      : Incident photon count (controls dose level). Default 1000.
        sigma_e : Electronic noise std (normalised). Default 0.002 ≈ 8 HU.

    Returns:
        Noisy image clipped to [0, 1].
    """
    lam = np.maximum(image * float(I0), 1.0)   # floor at 1: avoids Poisson(0)=0 always
    poisson_noisy = np.random.poisson(lam).astype(np.float32) / float(I0)
    electronic    = np.random.normal(loc=0.0, scale=sigma_e, size=image.shape).astype(np.float32)
    return np.clip(poisson_noisy + electronic, 0.0, 1.0)


def prep_dataset_standard_noise(config, I0=1000, sigma_e=0.002):
    """
    Build a synthetic CT dataset from full-dose DICOMs using the standard
    research-paper Poisson + Gaussian noise model.

    Input  (noisy) : full-dose slice + add_ct_noise_standard(I0, sigma_e)
    Target (clean) : original full-dose slice (ground truth)

    Saves to:
        <standard_noise_save_dir>/<slice_thickness> <kernel>/
            {patient}_input_{slice_num:03d}.npy   ← noisy
            {patient}_target_{slice_num:03d}.npy  ← clean

    Expected body-ROI PSNR (mask>0.02, 256×256):
        I0=1000, sigma_e=0.002  →  ~33–36 dB   (default, challenging)
        I0=2500, sigma_e=0.002  →  ~38–41 dB   (quarter-dose equivalent)
        I0=5000, sigma_e=0.001  →  ~42–44 dB   (half-dose equivalent)
    """
    save_dir = os.path.join(
        config['standard_noise_save_dir'],
        config['slice_thickness'] + " " + config['reconstruction_kernel']
    )
    os.makedirs(save_dir, exist_ok=True)
    print(f"Output dir  : {save_dir}")
    print(f"Noise params: I0={I0}, sigma_e={sigma_e}")

    data_dir = os.path.join(
        config['data_dir'],
        config['slice_thickness'] + " " + config['reconstruction_kernel']
    )
    full_dose_dir = os.path.join(
        data_dir, "FD_" + config['slice_thickness'], "full_" + config['slice_thickness']
    )

    for patient in sorted(os.listdir(full_dose_dir)):
        patient_path = os.path.join(
            full_dose_dir, patient, "full_" + config['slice_thickness']
        )
        if not os.path.exists(patient_path):
            print(f"  Skipping (not found): {patient_path}")
            continue

        all_slices = HU_converted(load_scan(patient_path))

        for slice_num, raw_slice in enumerate(all_slices):
            target = normalize(raw_slice, config)
            noisy  = add_ct_noise_standard(target, I0=I0, sigma_e=sigma_e)
            np.save(os.path.join(save_dir, f"{patient}_target_{slice_num:03d}.npy"), target)
            np.save(os.path.join(save_dir, f"{patient}_input_{slice_num:03d}.npy"),  noisy)

        print(f"  {patient}: {len(all_slices)} slices processed")

    print("Dataset preparation complete.")


if __name__ == '__main__':
    with open('config/data_prep.json', 'r') as f:
        config = json.load(f)

    # prep_dataset(config)                  # real paired LDCT
    # prep_dataset_gaussian_noise(config)   # old heavy synthetic (unphysical)
    prep_dataset_standard_noise(config)     # standard paper Poisson+Gaussian




    

