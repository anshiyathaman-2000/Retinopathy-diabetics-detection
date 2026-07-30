import cv2
import numpy as np

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    to the L-channel of the image in LAB color space.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def single_scale_retinex(img, sigma):
    """
    Applies Single Scale Retinex (SSR) to an image channel.
    """
    img_double = np.float64(img) + 1.0  # Prevent log(0)
    # Apply Gaussian blur
    blur = cv2.GaussianBlur(img_double, (0, 0), sigma)
    # SSR formula
    ssr = np.log10(img_double) - np.log10(blur + 1.0)
    return ssr

def multi_scale_retinex(img, sigmas=[5, 15, 30]):
    """
    Applies Multi-Scale Retinex (MSR) to a color BGR image.
    Uses scales appropriate for resized images.
    """
    msr = np.zeros(img.shape, dtype=np.float64)
    # Compute for each channel
    for c in range(3):
        channel = img[:, :, c]
        channel_msr = np.zeros(channel.shape, dtype=np.float64)
        for sigma in sigmas:
            channel_msr += single_scale_retinex(channel, sigma)
        channel_msr = channel_msr / len(sigmas)
        msr[:, :, c] = channel_msr

    # Restore color / normalize
    for c in range(3):
        c_min = np.min(msr[:, :, c])
        c_max = np.max(msr[:, :, c])
        if c_max != c_min:
            msr[:, :, c] = (msr[:, :, c] - c_min) / (c_max - c_min) * 255.0
        else:
            msr[:, :, c] = np.zeros(msr[:, :, c].shape)

    return np.uint8(np.clip(msr, 0, 255))

def preprocess_fundus_image(img_path, target_size=(256, 256), cache_dir=None):
    """
    Complete preprocessing pipeline for fundus images with disk caching:
    1. Check if cached version exists in cache directory
    2. If yes, load and return it (takes <1ms)
    3. If no, read BGR image, resize, CLAHE, MSR, save to cache, and return
    """
    import os
    if cache_dir is None:
        cache_dir = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/preprocessed_cache"
    os.makedirs(cache_dir, exist_ok=True)
    
    # Prepend the parent directory name (e.g. DR or No_DR) to avoid collisions
    parent_folder = os.path.basename(os.path.dirname(img_path))
    base_name = os.path.basename(img_path)
    cache_fn = f"{target_size[0]}_{target_size[1]}_{parent_folder}_{base_name}"
    cache_path = os.path.join(cache_dir, cache_fn)
    
    if os.path.exists(cache_path):
        cached_img = cv2.imread(cache_path)
        if cached_img is not None:
            return cached_img
            
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image at {img_path}")
    
    # 1. Resize first
    resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
    
    # 2. Contrast enhancement via CLAHE
    enhanced = apply_clahe(resized)
    
    # 3. Illumination normalization via MSR
    normalized = multi_scale_retinex(enhanced)
    
    # Save to cache
    cv2.imwrite(cache_path, normalized)
    
    return normalized



if __name__ == "__main__":
    # Test script on a sample validation image
    import os
    img_dir = "/Users/apple/Library/Mobile Documents/com~apple~CloudDocs/paper/Evaluation_Set/Validation"
    out_dir = "/Users/apple/.gemini/antigravity-ide/brain/3f945eae-5a3f-461b-9848-6694cbb5cbf7/scratch"
    
    os.makedirs(out_dir, exist_ok=True)
    images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
    if len(images) > 0:
        sample_path = os.path.join(img_dir, images[0])
        print(f"Testing preprocessing on: {sample_path}")
        try:
            processed = preprocess_fundus_image(sample_path)
            out_path = os.path.join(out_dir, "preprocessed_sample.png")
            cv2.imwrite(out_path, processed)
            print(f"Success! Preprocessed image saved to: {out_path}")
        except Exception as e:
            print(f"Error during preprocessing: {e}")
    else:
        print("No images found for testing.")
