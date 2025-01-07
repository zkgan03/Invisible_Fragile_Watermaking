import streamlit as st
import cv2
import numpy as np
import pywt
import pydicom
import time
import math
import hashlib
from math import floor
import matplotlib.pyplot as plt
import io
from skimage import img_as_float, img_as_ubyte
from skimage.filters import gaussian, median
from skimage.util import random_noise
from skimage.transform import resize
from skimage.io import imsave, imread
from skimage import exposure
from skimage.metrics import structural_similarity as ssim
from scipy.fftpack import dct, idct
from io import BytesIO
from PIL import Image
from skimage.metrics import mean_squared_error

## FUNCTIONS FOR DWT-SVD
def embed_image(img, wm, q, wavelet="haar"):
    """Embeds a watermark into an image using DWT and SVD.

    Args:
        img (numpy.ndarray): The original image (YCrCb color space).
        wm (numpy.ndarray): The watermark image (grayscale).
        q (int): The watermark strength factor.
        wavelet (str, optional): The wavelet type for DWT. Defaults to 'haar'.

    Returns:
        numpy.ndarray: The watermarked image (BGR color space).
    """

    # Extract Y (luminance) component and perform DWT
    LL, (HL, LH, HH) = pywt.dwt2(img, wavelet)
    LL_embedded = embed_subband(LL, wm, q)
    img_embedded = pywt.idwt2((LL_embedded, (HL, LH, HH)), wavelet)
    return img_embedded


def embed_subband(split, wm, Q):
    """Embeds watermark into a DWT subband using SVD.

    Args:
        split (numpy.ndarray): The DWT subband.
        wm (numpy.ndarray): The watermark image.
        Q (int): The watermark strength factor.

    Returns:
        numpy.ndarray: The watermarked subband.
    """

    h, w = split.shape
    for i in range(h // 4):
        for j in range(w // 4):
            u, s, v = np.linalg.svd(
                np.float32(split[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4])
            )
            s_max = int(s[0])
            a = np.mod(s_max, Q)

            if wm[i, j] == 0:
                s_max = s_max - a + (Q // 4 if 0 <= a < 3 * Q / 4 else 5 * Q // 4)
            elif wm[i, j] == 255:
                s_max = s_max - a - (Q // 4 if 0 <= a < Q // 4 else -3 * Q // 4)

            s[0] = s_max
            split[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4] = np.dot(
                np.dot(u, np.diag(s)), v
            )
    return split


def extract_watermark_DWT_SVD(img, q, wavelet="haar"):
    """Extracts a watermark from a DICOM image using DWT and SVD.

    Args:
        img (numpy.ndarray): The watermarked DICOM image.
        q (int): The watermark strength factor (used during embedding).
        wavelet (str, optional): The wavelet type used for DWT during embedding. Defaults to 'haar'.

    Returns:
        numpy.ndarray: The extracted watermark image (binary).
    """
    LL, _ = pywt.dwt2(
        img, wavelet
    )  # Directly use the DICOM image (no Y channel extraction)
    extracted_wm = extract_from_subband(LL, q)
    return extracted_wm


def extract_from_subband(split, Q):
    """Extracts watermark from a DWT subband using SVD.

    Args:
        split (numpy.ndarray): The DWT subband.
        Q (int): The watermark strength factor.

    Returns:
        numpy.ndarray: The extracted watermark.
    """

    h, w = split.shape
    extracted_wm = []
    for i in range(h // 4):
        for j in range(w // 4):
            _, s, _ = np.linalg.svd(
                np.float32(split[i * 4 : (i + 1) * 4, j * 4 : (j + 1) * 4])
            )
            a = np.mod(s[0], Q)
            extracted_wm.append(1 if a > Q / 2 else 0)
    return np.array(extracted_wm).reshape((64, 64))  # Assuming 64x64 watermark


# ==============================================================================
# SAVE & LOAD FUNCTIONS
# Function to load a DICOM image
def load_dicom(path):
    ds = pydicom.dcmread(path)
    image = ds.pixel_array
    return image, ds


def save_dicom(path, image, original_dicom):
    original_dicom.PixelData = image.astype(np.uint16).tobytes()
    original_dicom.save_as(path)


# =============================================================================
# FUNCTIONS FOR DWT with key (addtional functions from DWT-DCT)
def generate_key(string):
    """Generates a secret key from the watermark string using SHA-256.

    Args:
      watermark_string: The secret string to use for key generation.

    Returns:
      A bytes object representing the secret key.
    """
    # Encode the string to bytes and get the digest (key)
    key_bytes = hashlib.sha256(string.encode()).digest()

    # Convert the key bytes to a NumPy array of uint8 (unsigned 8-bit integers)
    secret_key = np.frombuffer(key_bytes, dtype=np.uint8)
        
    return secret_key

# Lower alpha value for less impact
def embed_watermark_with_key(orig_image, watermark, key, alpha = 0.00001):
  
    #ravel(): return a contiguous flattened 1D array
    height, width = orig_image.shape
    resized_watermark = cv2.resize(watermark, (height//16, width//16))
    watermark_flat = resized_watermark.ravel()
    
    ## resize generated_key to same length as watermark_flat
    generated_key = generate_key(key)
    generated_key = np.tile(generated_key, len(watermark_flat) // len(generated_key) + 1)[:len(watermark_flat)]
    
    # encryption
    encrypted_watermark = np.bitwise_xor(watermark_flat, generated_key).astype(np.uint8)
#     print("embed_watermark :",encrypted_watermark)

    # apply dct
    dct_ori_img = apply_dct(orig_image)
    ind = 0
    for y in range (0, height, 16):
        for x in range (0, width,16):
            if ind < len(watermark_flat):
                # get the sub dct
                subdct = dct_ori_img[y:y+16, x:x+16]
                subdct[15][15] = encrypted_watermark[ind] * alpha
                dct_ori_img[y:y+16, x:x+16] = subdct
                ind += 1
      
    #inverse dct
    idct_img = inverse_dct(dct_ori_img)
        
    return idct_img

def extract_watermark_with_key(orig_image, key, alpha = 0.00001):
   
    # Pre-allocate for watermark
    height, width = orig_image.shape
    encrypted_watermark = np.zeros((width//16) **2)  

    #apply dct
    dct_watermarked_img = apply_dct(orig_image) 
    
    ind = 0
    for y in range (0, height, 16):
        for x in range (0, width, 16):
            if ind < len(encrypted_watermark):
                subdct = dct_watermarked_img[y:y+16, x:x+16]
                encrypted_watermark[ind] = subdct[15][15] / alpha
                ind += 1

    ##round the value first
    ##some of the the value will get like 10.99999 or 10.00001
    encrypted_watermark = np.round(encrypted_watermark).astype(np.uint8)
#     print("extract_watermark :", encrypted_watermark)

    ## resize generated_key to same length as watermark_flat
    generated_key = generate_key(key)
    generated_key = np.tile(generated_key, len(encrypted_watermark) // len(generated_key) + 1)[:len(encrypted_watermark)]
    
    watermark = np.bitwise_xor(encrypted_watermark, generated_key).astype(np.uint8)
    
    # Reshape the extracted watermark to its original size
    watermark = watermark.reshape(height // 16, width // 16)
    

    return watermark
# =============================================================================
## FUNCTIONS FOR DWT-DCT
# Function to apply DCT to an image array
def apply_dct(image_array):
    size = len(image_array[0])  # get width of the image
    all_subdct = np.empty((size, size))  # create an empty array for DCT image
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subpixels = image_array[i : i + 8, j : j + 8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i : i + 8, j : j + 8] = subdct
    return all_subdct


# Function to inverse DCT on the DCT array
def inverse_dct(all_subdct):
    size = len(all_subdct[0])
    all_subidct = np.empty((size, size))
    for i in range(0, size, 8):
        for j in range(0, size, 8):
            subidct = idct(
                idct(all_subdct[i : i + 8, j : j + 8].T, norm="ortho").T, norm="ortho"
            )
            all_subidct[i : i + 8, j : j + 8] = subidct
    return all_subidct


# Function to embed a watermark into an image
# def embed_watermark(watermark_array, orig_image, alpha=0.00001):
#     height, width = orig_image.shape
#     resized_watermark = cv2.resize(watermark_array, (height // 16, width // 16))
#     dct_ori_img = apply_dct(orig_image)

#     # ravel(): return a contiguous flattened 1D array
#     watermark_flat = resized_watermark.ravel()
#     ind = 0
#     for y in range(0, height, 16):
#         for x in range(0, width, 16):
#             if ind < len(watermark_flat):
#                 subdct = dct_ori_img[y : y + 16, x : x + 16]
#                 subdct[15][15] = watermark_flat[ind] * alpha
#                 dct_ori_img[y : y + 16, x : x + 16] = subdct
#                 ind += 1
#     idct_img = inverse_dct(dct_ori_img)
#     return idct_img

def embed_watermark_DWT_DCT(orig_image, watermark, key = None, alpha = 0.00001):
  
    # apply dwt, then dct
    LL, (HL, LH, HH) = apply_dwt(orig_image)
    dct_img = apply_dct(LL)
    
    #ravel(): return a contiguous flattened 1D array
    height, width = LL.shape
    resized_watermark = cv2.resize(watermark, (height//16, width//16))
    watermark_flat = resized_watermark.ravel()
    
    ## resize generated_key to same length as watermark_flat
    generated_key = generate_key(key)
    generated_key = np.tile(generated_key, len(watermark_flat) // len(generated_key) + 1)[:len(watermark_flat)]
    
    # encryption
    encrypted_watermark = np.bitwise_xor(watermark_flat, generated_key).astype(np.uint8)
    
    ind = 0
    for y in range (0, height, 16):
        for x in range (0, width,16):
            if ind < len(watermark_flat):
                # get the sub dct
                subdct = dct_img[y:y+16, x:x+16]
                subdct[15][15] = encrypted_watermark[ind] * alpha
                dct_img[y:y+16, x:x+16] = subdct
                ind += 1
      
    #inverse dct
    idct_img = inverse_dct(dct_img)
    
    #inverse dwt
    idwt_img = inverse_dwt((idct_img, (HL, LH, HH)))
        
    return idwt_img


# Function to extract watermark from an watermarked image
def extract_watermark_DWT_DCT(orig_image, key = None, alpha = 0.00001):
    # apply dwt, then dct
    LL, (HL, LH, HH) = apply_dwt(orig_image)
    dct_watermarked_img = apply_dct(LL)
    
    # Pre-allocate for watermark
    height, width = LL.shape
    encrypted_watermark = np.zeros((width//16) **2)  
    
    ind = 0
    for y in range (0, height, 16):
        for x in range (0, width, 16):
            if ind < len(encrypted_watermark):
                subdct = dct_watermarked_img[y:y+16, x:x+16]
                encrypted_watermark[ind] = subdct[15][15] / alpha
                ind += 1

    ##round the value first
    ##some of the the value will get like 10.99999 or 10.00001
    encrypted_watermark = np.round(encrypted_watermark).astype(np.uint8)
#     print("extract_watermark :", encrypted_watermark)

    ## resize generated_key to same length as watermark_flat
    generated_key = generate_key(key)
    generated_key = np.tile(generated_key, len(encrypted_watermark) // len(generated_key) + 1)[:len(encrypted_watermark)]
    
    watermark = np.bitwise_xor(encrypted_watermark, generated_key).astype(np.uint8)
    
    # Reshape the extracted watermark to its original size
    watermark = watermark.reshape(height // 16, width // 16)
    

    return watermark


def apply_dwt(img, wavelet="haar"):
    return pywt.dwt2(img, wavelet)


def inverse_dwt(dwt, wavelet="haar"):
    LL, (HL, LH, HH) = dwt
    return pywt.idwt2((LL, (HL, LH, HH)), wavelet)


# ===============================================================================
# FUNCTIONS FOR LSB
def generate_circular_mask(img):
    # 1. get length `m` and width `n` of input img
    n = img.shape[0]
    m = img.shape[1]
#     print(n,m)

    # 2. divide `m` into equal 16 parts and call each unit as `lu`
    lu = m/16
#     print(lu)

    # 3. divide `n` into equal 16 parts and call each unit as `wu`
    wu = n/16
#     print(wu)

    # 4. `a` = (`lu`*14) /2    (lu*14 will get the major axis, to get semi major axis of ellipse,, need to divide by 2)
    a = int(lu * 14 /2)
#     print(a)

    # 5. `b` = (`wu`*12) /2    (wu*12 will get the minor axis, to get semi minor axis of ellipse, need to divide by 2)
    b = int(wu * 12 /2)
#     print(b)

    circular_mask = np.zeros((n,m), dtype=np.uint8)
#     print(circular_mask.shape)

    cv2.ellipse(circular_mask, (m // 2, n // 2), (a, b), 0, 0, 360, (255, 255, 255),thickness = -1)
#     cv2.imshow("test", circular_mask)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    ## input img : 256*256
    ## a=112, b=96, h=128, k=128

    return circular_mask

def segmentation(img, mask):

    mask_inv = cv2.bitwise_not(mask)

    roi = cv2.bitwise_and(img,img, mask=mask)
    roni = cv2.bitwise_and(img,img, mask=mask_inv)

    return (roi, roni)


def embed_watermark_LSB(orig_image, watermark, key=None):
    watermark = cv2.resize(watermark, (orig_image.shape[0], orig_image.shape[1]))

    ## Ensure watermark in binary
    watermark = (watermark > 128).astype(np.uint8)  # Convert to binary (0 or 1)

    ## use mask to get roi
    mask = generate_circular_mask(orig_image)
    mask = (mask > 128).astype(np.uint8)  # Convert to binary (0 or 1)

    ## change to 16bits, to retain img data
    mask_16 = mask.astype(np.uint16)
    watermark_16 = watermark.astype(np.uint16)

    # use mask, to embed in ROI only
    ## interprete from left to right
    ## img_roi & ~mask --> if it is px needed to be embed(mask=1), turn LSB to 0 (thats why ~mask)
    ## watermark & mask --> if it is px needed to be embed (mask=1), the intensity value depends on watermark
    watermarked_img = orig_image & ~mask_16 | (watermark_16 & mask_16)

    return watermarked_img


def extract_watermark_LSB(orig_image, key=None):

    ## use mask to get roi
    mask = generate_circular_mask(orig_image)
    mask = (mask > 128).astype(np.uint8)  # Convert to binary (0 or 1)

    # Extract LSB where mask is 1
    extracted_bin_img = orig_image.astype(np.uint16) & mask.astype(np.uint16)

    # Convert to proper pixel values for viewing (scale to 0-255)
    extracted_watermark = extracted_bin_img * 255

    return extracted_watermark

# ===============================================================================
# ATTACK FUNCTIONS
def apply_attacks_and_extract(dicom_image, extract_function, attack_level):
    image = img_as_float(
        dicom_image.copy()
    )  # Ensure image is in the right format for attacks
    attacked_images = apply_attacks(image, attack_level)
    extracted_watermarks = {}
    list_of_BER = {}

    for name, attacked_img in attacked_images.items():
        extracted_watermarks[name] = extract_function(img_as_float(attacked_img), 50)

    num_attacks = len(attacked_images)
    fig, axs = plt.subplots(num_attacks, 2, figsize=(10, 2 * num_attacks))

    for i, (attack_name, attacked_img) in enumerate(attacked_images.items()):
        axs[i, 0].imshow(attacked_img, cmap="gray")
        axs[i, 0].set_title(f"{attack_name} - Attacked")
        axs[i, 0].axis("off")

        ber = calculate_ber(extract_function(image), extracted_watermarks[attack_name])
        list_of_BER[attack_name] = ber
        st.write(f"BER for {attack_name}: {ber}")

        if extracted_watermarks.get(attack_name) is not None:
            axs[i, 1].imshow(extracted_watermarks[attack_name], cmap="gray")
            axs[i, 1].set_title(f"{attack_name} - Extracted")
            axs[i, 1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


def apply_attacks_and_extract_with_key(dicom_image, extract_function, key, attack_level):
    image = img_as_float(dicom_image.copy())  # Ensure image is in the right format for attacks
    attacked_images = apply_attacks(image, attack_level)
    extracted_watermarks = {}
    list_of_BER = {}
    for name, attacked_img in attacked_images.items():
        extracted_watermarks[name] = extract_function(img_as_float(attacked_img), key)
    num_attacks = len(attacked_images)
    fig, axs = plt.subplots(num_attacks, 2, figsize=(10, 2 * num_attacks))
    for i, (attack_name, attacked_img) in enumerate(attacked_images.items()):
        axs[i, 0].imshow(attacked_img, cmap="gray")
        axs[i, 0].set_title(f"{attack_name} - Attacked")
        axs[i, 0].axis("off")

        ber = calculate_ber(extract_function(img_as_float(dicom_image), key), extracted_watermarks[attack_name])
        list_of_BER[attack_name] = ber
        st.write(f"BER for {attack_name}: {ber}")

        if extracted_watermarks.get(attack_name) is not None:
            axs[i, 1].imshow(extracted_watermarks[attack_name], cmap="gray")
            axs[i, 1].set_title(f"{attack_name} - Extracted")
            axs[i, 1].axis("off")
    plt.tight_layout()
    st.pyplot(fig)


def apply_attacks(image, attack_level):
    attacks = {
        "Without Attack": lambda x: x,
        "Minimal Gaussian Blur": lambda x: gaussian(x, sigma=1 * attack_level),
        "Moderate Gaussian Blur": lambda x: gaussian(x, sigma=2 * attack_level),
        "Median Filtering": lambda x: median(x),
        "JPEG Compression - Low": lambda x: jpeg_compression(
            x, quality=attack_level + 35
        ),
        "JPEG Compression - High": lambda x: jpeg_compression(x, quality=attack_level),
        "Scaling Attack - Down & Up": lambda x: scale_down_up(x),
        "Very Light Smoothing": lambda x: gaussian(x, sigma=0.5 * attack_level),
    }
    attacked_images = {}
    for name, attack_func in attacks.items():
        attacked_images[name] = attack_func(image.copy())
    return attacked_images


def jpeg_compression(dicom_image, quality=85):
    # Check if the image is already in 8-bit format, if not, normalize and convert
    if dicom_image.dtype != np.uint8:
        # Normalize the image to 0-255 range
        max_value = np.max(dicom_image)
        min_value = np.min(dicom_image)
        normalized_image = 255 * (dicom_image - min_value) / (max_value - min_value)
        image_8bit = normalized_image.astype(np.uint8)
    else:
        image_8bit = dicom_image

    # Set JPEG quality level for compression
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encoded_image = cv2.imencode(".jpg", image_8bit, encode_param)
    if result:
        decoded_image = cv2.imdecode(encoded_image, cv2.IMREAD_UNCHANGED)

        # Check if color conversion is necessary (if original was color, convert back to grayscale)
        if len(decoded_image.shape) == 3 and decoded_image.shape[2] == 3:
            decoded_image = cv2.cvtColor(decoded_image, cv2.COLOR_BGR2GRAY)
        return decoded_image
    else:
        raise Exception("Image encoding failed.")


def scale_down_up(image, scale_factor=0.5):
    height, width = image.shape[:2]
    resized_down = resize(
        image,
        (int(height * scale_factor), int(width * scale_factor)),
        anti_aliasing=True,
    )
    resized_up = resize(resized_down, (height, width), anti_aliasing=True)
    return resized_up


# Attack function for LSB
def apply_attacks_and_extract_LSB(img, extract_function):

    attacks = {
        "Without Attack": lambda x: x,  # No attack on the image
        "Minimal Gaussian Blur": lambda x: cv2.GaussianBlur(x, (3, 3), 0),  # Very light Gaussian blur
        "Median Filtering": lambda x: cv2.medianBlur(x, 3),  # Light median filter
        "JPEG Compression": lambda x: jpeg_compression_LSB(x, quality=85),  # Slight compression
        "Scaling Attack": lambda x: scale_down_up_LSB(x),
        "Very Light Smoothing": lambda x: cv2.blur(x, (2, 2))  # Very slight smoothing
    }

    print("Extracting watermark from the original image")

    # Initialize plotting
    num_attacks = len(attacks)
    fig, axs = plt.subplots(num_attacks, 2, figsize=(10, 2 * num_attacks))  # 2 columns per attack

    for i, (attack_name, attack_function) in enumerate(attacks.items()):
        # Apply the attack
        attacked_img = attack_function(img.copy())
        # Extract data using the provided extraction function
        extracted_img = extract_function(attacked_img)
        # Show the attacked image
        axs[i, 0].imshow(attacked_img, cmap='gray')
        axs[i, 0].set_title(f"{attack_name} - Attacked")
        axs[i, 0].axis('off')
        # Show the extracted image
        axs[i, 1].imshow(extracted_img, cmap='gray')
        axs[i, 1].set_title(f"{attack_name} - Extracted")
        axs[i, 1].axis('off')

        ber = calculate_ber(extract_function(img), extracted_img)

        st.write(f"BER for {attack_name}: {ber}")

    plt.tight_layout()
    st.pyplot(fig)


def jpeg_compression_LSB(image, quality=90):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return cv2.cvtColor(decimg, cv2.COLOR_BGR2GRAY)

def scale_down_up_LSB(image, scale_factor=0.5):
    height, width = image.shape
    resized_down = cv2.resize(image, (int(width * scale_factor), int(height * scale_factor)), interpolation=cv2.INTER_LINEAR)
    resized_up = cv2.resize(resized_down, (width, height), interpolation=cv2.INTER_LINEAR)
    return resized_up


def apply_attacks_and_extract_SVD(dicom_image, extract_function, attack_level, Q):
    image = img_as_float(
        dicom_image.copy()
    )  # Ensure image is in the right format for attacks
    attacked_images = apply_attacks(image, attack_level)
    extracted_watermarks = {}
    list_of_BER = {}

    for name, attacked_img in attacked_images.items():
        extracted_watermarks[name] = extract_function(img_as_float(attacked_img), Q)

    num_attacks = len(attacked_images)
    fig, axs = plt.subplots(num_attacks, 2, figsize=(10, 2 * num_attacks))

    for i, (attack_name, attacked_img) in enumerate(attacked_images.items()):
        axs[i, 0].imshow(attacked_img, cmap="gray")
        axs[i, 0].set_title(f"{attack_name} - Attacked")
        axs[i, 0].axis("off")

        ber = calculate_ber(extract_function(image, Q), extracted_watermarks[attack_name])
        list_of_BER[attack_name] = ber
        st.write(f"BER for {attack_name}: {ber}")

        if extracted_watermarks.get(attack_name) is not None:
            axs[i, 1].imshow(extracted_watermarks[attack_name], cmap="gray")
            axs[i, 1].set_title(f"{attack_name} - Extracted")
            axs[i, 1].axis("off")

    plt.tight_layout()
    st.pyplot(fig)


# ====================================================================================
# Function to calculate NC, PSNR, SSIM, and BER
from skimage.metrics import (
    normalized_root_mse,
    peak_signal_noise_ratio,
    structural_similarity,
)
from skimage.transform import resize

def calculate_metrics(original_img, watermarked_img, extracted_wm, watermark_img):
    # Normalized Cross-Correlation
    nc = np.corrcoef(original_img.ravel(), watermarked_img.ravel())[0, 1]
    # Peak Signal-to-Noise Ratio
    psnr = cv2.PSNR(original_img.astype(np.float32), watermarked_img.astype(np.float32))
    # Structural Similarity Index
    ssim_val = structural_similarity(
        original_img,
        watermarked_img,
        data_range=original_img.max() - original_img.min(),
    )

    # Resize the extracted_wm array to match watermark_img shape
    extracted_wm_resized = resize(extracted_wm, watermark_img.shape)

    # Bit Error Rate
    bits_error = np.sum(np.abs(extracted_wm_resized - watermark_img)) / (
        watermark_img.shape[0] * watermark_img.shape[1]
    )

    return nc, psnr, ssim_val, bits_error


def calculate_ber(original_watermark, extracted_watermark):
    # Calculate the number of bits in the watermark image
    total_bits = original_watermark.shape[0] * original_watermark.shape[1]
    # Count the number of bits that are different between the original and extracted watermarks
    error_bits = np.sum(original_watermark != extracted_watermark)
    # Calculate the Bit Error Rate
    ber = error_bits / total_bits
    return ber

# ======================================================================================
from PIL import Image, ImageDraw, ImageFont

def create_binary_watermark(text_lines, font_path="arial.ttf", font_size=10):
    """Creates a 64x64 binary watermark image with multiple lines of text.

    Args:
        text_lines (list): A list of strings representing the lines of text.
        font_path (str): Path to the TrueType font file.
        font_size (int): Font size in pixels.

    Returns:
        numpy.ndarray: A 64x64 NumPy array representing the binary watermark image.
    """

    img = Image.new('1', (64, 64), 0)  
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, font_size)
    text_color = 1 

    # Calculate line spacing and starting position
    ascent, descent = font.getmetrics()
    line_height = ascent + descent
    line_spacing = line_height + 0  # Add extra spacing for visual clarity
    y_text = (64 - len(text_lines) * line_spacing) // 2

    # Draw each line of text
    for line in text_lines:
        text_width = font.getlength(line)   
        x_text = (64 - text_width) // 2 
        draw.text((x_text, y_text), line, font=font, fill=text_color)
        y_text += line_spacing
        
    img.save("wm.png")
    return np.array(img)



def main():
    ## Idenfiy variable
    OUTPUT_DIR = "./watermarked_image/"
    IMG_NAME = "medical-img.dcm"

    if "watermarked_image" not in st.session_state:
        st.session_state["watermarked_image"] = None

    # Tittle Of The Program
    st.title("DICOM Watermarking and Extraction")

    # Dropdown Box to Select Algo
    method = st.sidebar.selectbox(
        "Select Watermarking Algorithm", ["DWT-DCT with key", "DCT with key", "DWT-SVD", "LSB"]
    )

    # Input of Attack Level to Watermarked image
    attack_level = st.sidebar.slider(
        "Level of Attack", min_value=1, max_value=100, value=50, step=1
    )

    # Input of key to Watermarked image
    key = st.sidebar.text_input("Key")

    # For User Upload Images
    dicom_img_file = st.file_uploader("Upload DICOM File", type=["dcm"])
    watermark_img_file = st.file_uploader("Upload Watermark Image", type=["png", "jpg"])

    if method == "DWT-DCT with key":
        if dicom_img_file is not None and watermark_img_file is not None:

            st.title("Results:")
            dicom_img, ds = load_dicom(dicom_img_file)

            # Convert the uploaded watermark image to a bytes object
            watermark_bytes = watermark_img_file.read()
            # Convert the bytes object to a numpy array
            watermark_np = np.frombuffer(watermark_bytes, np.uint8)
            # Decode the numpy array using cv2.imdecode
            watermark = cv2.imdecode(watermark_np, cv2.IMREAD_GRAYSCALE)
            watermark = (watermark > 128).astype(np.uint8)  # Convert to binary (0 or 1)

            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            # Display the original DICOM image
            axs.imshow(dicom_img, cmap="gray")
            axs.set_title("Original DICOM Image")
            axs.axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            # Do Watermarking
            watermarked_img = embed_watermark_DWT_DCT(dicom_img, watermark, key)

            # Buttons Click for Different Functions
            # Button - Display Embed Watermark Images & PSNR
            if st.sidebar.button("Embed Watermark"):
                st.subheader("Result:")
                # save it to session
                st.session_state["watermarked_image"] = watermarked_img
                # Diplay Watermarked image
                st.subheader("Watermarked Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(watermarked_img, cmap="gray")
                axs.set_title("Embed Watermark Image")
                axs.axis("off")
                st.pyplot(fig)

                # Calculate and display metrics
                extracted_watermark = extract_watermark_DWT_DCT(watermarked_img, key)
                nc, psnr, ssim_val, ber = calculate_metrics(
                    dicom_img, watermarked_img, extracted_watermark, watermark
                )
                st.write(f"Normalized Cross-Correlation: {nc}")
                st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
                st.write(f"Structural Similarity Index (SSIM): {ssim_val}")
                # st.write(f"Bit Error Rate (BER): {ber}")

            # Button - Extract & Display Watermark from embeded Watermark Image
            if st.sidebar.button("Extract Watermark"):
                st.subheader("Result:")
                extracted_watermark = extract_watermark_DWT_DCT(st.session_state["watermarked_image"], key)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_watermark, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Save Watermarked Image
            if st.sidebar.button("Save Watermarked Image"):

                IMG_NAME = "medical-img-by-DCT.dcm" # Set the file name 
                st.subheader("Result:")

                # This can make the picture show like original, but mcm will broke the watermark
                watermarked_img = np.clip(
                    watermarked_img, ds.pixel_array.min(), ds.pixel_array.max()
                )
                save_dicom(OUTPUT_DIR + IMG_NAME, watermarked_img, ds)
                st.subheader("Watermarked Image Saved")
                st.write(f"Image have been saved to {OUTPUT_DIR + IMG_NAME} .")

            # Button - Extract & Display Watermark from Original Uploaded Image
            if st.sidebar.button("Extract Watermark from original image"):
                st.subheader("Result:")
                extracted_wm = extract_watermark_DWT_DCT(dicom_img, key)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_wm, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Display A Series Type of Attack and the Result for Image and Watermark
            if st.sidebar.button("Show Attack Effect of the image and watermark"):
                st.subheader("Result:")
                apply_attacks_and_extract_with_key(
                    watermarked_img, extract_watermark_DWT_DCT, key, attack_level 
                )

    if method == "DCT with key":
        if dicom_img_file is not None and watermark_img_file is not None:

            st.title("Results:")
            dicom_img, ds = load_dicom(dicom_img_file)

            # Convert the uploaded watermark image to a bytes object
            watermark_bytes = watermark_img_file.read()
            # Convert the bytes object to a numpy array
            watermark_np = np.frombuffer(watermark_bytes, np.uint8)
            # Decode the numpy array using cv2.imdecode
            watermark = cv2.imdecode(watermark_np, cv2.IMREAD_GRAYSCALE)
            watermark = (watermark > 128).astype(np.uint8)  # Convert to binary (0 or 1)

            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            # Display the original DICOM image
            axs.imshow(dicom_img, cmap="gray")
            axs.set_title("Original DICOM Image")
            axs.axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            # Do Watermarking
            watermarked_img = embed_watermark_with_key(dicom_img, watermark, key)

            # Buttons Click for Different Functions
            # Button - Display Embed Watermark Images & PSNR
            if st.sidebar.button("Embed Watermark"):
                st.subheader("Result:")
                # Save it to session
                st.session_state["watermarked_image"] = watermarked_img
                # Diplay Watermarked image
                st.subheader("Watermarked Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(watermarked_img, cmap="gray")
                axs.set_title("Embed Watermark Image")
                axs.axis("off")
                st.pyplot(fig)

                # Calculate and display metrics
                extracted_watermark = extract_watermark_with_key(watermarked_img, key)
                nc, psnr, ssim_val, ber = calculate_metrics(
                    dicom_img, watermarked_img, extracted_watermark, watermark
                )
                st.write(f"Normalized Cross-Correlation: {nc}")
                st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
                st.write(f"Structural Similarity Index (SSIM): {ssim_val}")
                # st.write(f"Bit Error Rate (BER): {ber}")

            # Button - Extract & Display Watermark from embeded Watermark Image
            if st.sidebar.button("Extract Watermark"):
                st.subheader("Result:")
                extracted_watermark = extract_watermark_with_key(
                    st.session_state["watermarked_image"], key
                )
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_watermark, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Save Watermarked Image
            if st.sidebar.button("Save Watermarked Image"):
                IMG_NAME = "medical-img-by-DCT with key.dcm"  # Set the file name
                st.subheader("Result:")

                # This can make the picture show like original, but mcm will broke the watermark
                watermarked_img = np.clip(
                    watermarked_img, ds.pixel_array.min(), ds.pixel_array.max()
                )
                save_dicom(OUTPUT_DIR + IMG_NAME, watermarked_img, ds)
                st.subheader("Watermarked Image Saved")
                st.write(f"Image have been saved to {OUTPUT_DIR + IMG_NAME} .")

            # Button - Extract & Display Watermark from Original Uploaded Image
            if st.sidebar.button("Extract Watermark from original image"):
                st.subheader("Result:")
                extracted_wm = extract_watermark_with_key(dicom_img, key)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_wm, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Display A Series Type of Attack and the Result for Image and Watermark
            if st.sidebar.button("Show Attack Effect of the image and watermark"):
                st.subheader("Result:")
                apply_attacks_and_extract_with_key(
                    watermarked_img, extract_watermark_with_key, key, attack_level
                )

    if method == "DWT-SVD":
        if dicom_img_file and watermark_img_file:

            # Slider for Input
            Q = 50

            dicom_img, ds = load_dicom(dicom_img_file)
            # Convert the uploaded watermark image to a bytes object
            watermark_bytes = watermark_img_file.read()
            # Convert the bytes object to a numpy array
            watermark_np = np.frombuffer(watermark_bytes, np.uint8)
            # Decode the numpy array using cv2.imdecode
            watermark = cv2.imdecode(watermark_np, cv2.IMREAD_GRAYSCALE)

            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            # Display the original DICOM image
            axs.imshow(dicom_img, cmap="gray")
            axs.set_title("Original DICOM Image")
            axs.axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            # Do Watermarking
            watermarked_img = embed_image(dicom_img, watermark, Q)
            watermarked_img = np.clip(
                watermarked_img, ds.pixel_array.min(), ds.pixel_array.max()
            )

            # Buttons Click for Different Functions
            # Button - Display Embed Watermark Images & PSNR
            if st.sidebar.button("Embed Watermark"):
                st.subheader("Result:")
                # Diplay Watermarked image
                st.subheader("Watermarked Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(watermarked_img, cmap="gray")
                axs.set_title("Embedded Watermark Image")
                axs.axis("off")
                st.pyplot(fig)

                # Calculate and display metrics
                extracted_watermark = extract_watermark_DWT_SVD(watermarked_img, Q)
                nc, psnr, ssim_val, ber = calculate_metrics(
                    dicom_img, watermarked_img, extracted_watermark, watermark
                )
                st.write(f"Normalized Cross-Correlation: {nc}")
                st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
                st.write(f"Structural Similarity Index (SSIM): {ssim_val}")
                # st.write(f"Bit Error Rate (BER): {ber}")

            # Button - Extract & Display Watermark from embeded Watermark Image
            if st.sidebar.button("Extract Watermark"):
                st.subheader("Result:")
                extracted_wm = extract_watermark_DWT_SVD(watermarked_img, Q)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_wm, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Save Watermarked Image
            if st.sidebar.button("Save Watermarked Image"):
                IMG_NAME = "medical-img-by-DWT-SVD.dcm"  # Set the file name

                st.subheader("Result:")
                save_dicom(OUTPUT_DIR + IMG_NAME, watermarked_img, ds)
                st.subheader("Watermarked Image Saved")
                st.write(f"Image have been saved to {OUTPUT_DIR + IMG_NAME} .")

            # Button - Extract & Display Watermark from Original Uploaded Image
            if st.sidebar.button("Extract Watermark from original image"):
                st.subheader("Result:")
                extracted_wm = extract_watermark_DWT_SVD(dicom_img, Q)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_wm, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Display A Series Type of Attack and the Result for Image and Watermark
            if st.sidebar.button("Show Attack Effect of the image and watermark"):
                st.subheader("Result:")
                apply_attacks_and_extract_SVD(
                    watermarked_img, extract_watermark_DWT_SVD, attack_level, Q
                )

    if method == "LSB":
        if dicom_img_file is not None and watermark_img_file is not None:

            st.title("Results:")
            dicom_img, ds = load_dicom(dicom_img_file)

            # Convert the uploaded watermark image to a bytes object
            watermark_bytes = watermark_img_file.read()
            # Convert the bytes object to a numpy array
            watermark_np = np.frombuffer(watermark_bytes, np.uint8)
            # Decode the numpy array using cv2.imdecode
            watermark = cv2.imdecode(watermark_np, cv2.IMREAD_GRAYSCALE)
            # watermark = (watermark > 128).astype(np.uint8)  # Convert to binary (0 or 1)

            fig, axs = plt.subplots(1, 1, figsize=(15, 5))
            # Display the original DICOM image
            axs.imshow(dicom_img, cmap="gray")
            axs.set_title("Original DICOM Image")
            axs.axis("off")

            plt.tight_layout()
            st.pyplot(fig)

            # Do Watermarking
            watermarked_img = embed_watermark_LSB(dicom_img, watermark, key)

            # Buttons Click for Different Functions
            # Button - Display Embed Watermark Images & PSNR
            if st.sidebar.button("Embed Watermark"):
                st.subheader("Result:")
                # Diplay Watermarked image
                st.subheader("Watermarked Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(watermarked_img, cmap="gray")
                axs.set_title("Embed Watermark Image")
                axs.axis("off")
                st.pyplot(fig)

                # Calculate and display metrics
                extracted_watermark = extract_watermark_LSB(watermarked_img, key)
                nc, psnr, ssim_val, ber = calculate_metrics(
                    dicom_img, watermarked_img, extracted_watermark, watermark
                )
                st.write(f"Normalized Cross-Correlation: {nc}")
                st.write(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
                st.write(f"Structural Similarity Index (SSIM): {ssim_val}")
                # st.write(f"Bit Error Rate (BER): {ber}")

            # Button - Extract & Display Watermark from embeded Watermark Image
            if st.sidebar.button("Extract Watermark"):
                st.subheader("Result:")
                extracted_watermark = extract_watermark_LSB(watermarked_img, key)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_watermark, cmap="gray")
                axs.set_title("Extracted Watermark")

                st.pyplot(fig)

            # Button - Save Watermarked Image
            if st.sidebar.button("Save Watermarked Image"):

                IMG_NAME = "medical-img-by-LSB.dcm"  # Set the file name
                st.subheader("Result:")

                # This can make the picture show like original, but mcm will broke the watermark
                # watermarked_img = np.clip(
                #     watermarked_img, ds.pixel_array.min(), ds.pixel_array.max()
                # )
                save_dicom(OUTPUT_DIR + IMG_NAME, watermarked_img, ds)
                st.subheader("Watermarked Image Saved")
                st.write(f"Image have been saved to {OUTPUT_DIR + IMG_NAME} .")

            # Button - Extract & Display Watermark from Original Uploaded Image
            if st.sidebar.button("Extract Watermark from original image"):
                st.subheader("Result:")
                extracted_wm = extract_watermark_LSB(dicom_img)
                st.subheader("Extracted Watermark Image")
                fig, axs = plt.subplots(1, 1, figsize=(15, 5))
                axs.imshow(extracted_wm, cmap="gray")
                axs.set_title("Extracted Watermark")
                axs.axis("off")
                st.pyplot(fig)

            # Button - Display A Series Type of Attack and the Result for Image and Watermark
            if st.sidebar.button("Show Attack Effect of the image and watermark"):
                st.subheader("Result:")
                apply_attacks_and_extract_LSB(
                    watermarked_img, extract_watermark_LSB
                )

if __name__ == "__main__":  
    main()
