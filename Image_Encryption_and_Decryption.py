# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 08:18:26 2024

@author: gopub
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:27:34 2024

@author: gopub
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

# Function that gives the standard logistic map equation
def logistic_map(x, r):
    return r * x * (1 - x)

# Function to apply logistic map encryption 
def encrypt_logistic_map(input_image, r=3.9, num_iterations=1000000):
    # Open the input image
    image = Image.open(input_image)
    width, height = image.size

    # Flatten the image pixel values #WRONG?
    pixels = list(image.getdata())

    # Initialize logistic map with a random seed
    x = 0.5

    # Encrypt the image using logistic map
    for _ in range(num_iterations):
        x = logistic_map(x, r)
        index = int((x * width * height) % (width * height))
        pixels[index], pixels[-index] = pixels[-index], pixels[index]

    # Create a new image with the encrypted pixel values
    encrypted_image = Image.new(image.mode, (width, height))
    encrypted_image.putdata(pixels)

    return encrypted_image


# Function to generate encryption key
def generate_key():
    # Generate a random password and salt
    password = os.urandom(16)  # Generate a random password
    salt = os.urandom(16)  # Generate a random salt

    # Derive a key using PBKDF2
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,  # Length of the key in bytes (256 bits)
        salt=salt,
        iterations=100000,  # Number of iterations (adjust as needed)
        backend=default_backend()
    )
    key = kdf.derive(password)
    return key


# Function to apply block cipher encryption using AES algorithm
def apply_block_cipher(input_image, key, iv):
    # Convert the image to bytes
    image_bytes = input_image.tobytes()

    # Pad the image bytes if needed
    padder = padding.PKCS7(128).padder()
    padded_data = padder.update(image_bytes) + padder.finalize()

    # Create an AES cipher with CBC mode
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Encrypt the padded image bytes
    encryptor = cipher.encryptor()
    encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
    
    # Calling a global variable to store the encrypted data bytes
    global encrypted_data_to_pass
    encrypted_data_to_pass = encrypted_data
    

    # Convert the modified bytes back to an image
    modified_image = Image.frombytes(input_image.mode, input_image.size, encrypted_data)
    return modified_image

# Function to apply stream cipher encryption (operation: XOR)
def apply_stream_cipher(input_image, key):
    
    # Convert the image to bytes
    image_in_bytes = input_image.tobytes()
    
    #Cinvert the bytes to byte array
    image_bytes = bytearray(image_in_bytes)

    # Perform XOR operation between each byte of the image and corresponding byte of the key
    for i in range(len(image_bytes)):
        image_bytes[i] ^= key[i % len(key)]

    # Convert the modified bytes back to an image
    modified_image = Image.frombytes(input_image.mode, input_image.size, bytes(image_bytes))
    return modified_image

# Function to apply stream cipher decryption (operation: XOR)
def apply_stream_cipher_decryption(encrypted_bytes, key):
    # Convert the encrypted bytes to a mutable bytearray
    encrypted_bytearray = bytearray(encrypted_bytes)

    # Perform XOR operation between each byte of the encrypted image and corresponding byte of the key
    for i in range(len(encrypted_bytearray)):
        encrypted_bytearray[i] ^= key[i % len(key)]

    # Convert the modified bytearray back to bytes
    decrypted_bytes = bytes(encrypted_bytearray)

    return decrypted_bytes


# Function to apply block cipher decryption using AES algorithm
def apply_block_cipher_decryption(encrypted_bytes, key, iv):
   

    # Create a decryption cipher object
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())

    # Create a decryptor object
    decryptor = cipher.decryptor()

    # Decrypt the encrypted image bytes
    decrypted_data = decryptor.update(encrypted_bytes) + decryptor.finalize()

    # Remove padding
    unpadder = padding.PKCS7(128).unpadder()
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()

    
    return unpadded_data

# Function to apply logistic map decryption 
def decrypt_logistic_map(encrypted_image, r=3.9, num_iterations=1000000):
    # Get the size of the encrypted image
    width, height = encrypted_image.size

    # Flatten the encrypted image pixel values
    pixels = list(encrypted_image.getdata())

    # Initialize logistic map with the same seed used for encryption
    x = 0.5

    # Decrypt the image using inverse logistic map
    for _ in range(num_iterations):
        x = logistic_map(x, r)
        index = int((x * width * height) % (width * height))
        pixels[index], pixels[-index] = pixels[-index], pixels[index]

    # Create a new image with the decrypted pixel values
    decrypted_image = Image.new(encrypted_image.mode, (width, height))
    decrypted_image.putdata(pixels)

    return decrypted_image


# Function to calculate NPCR (Normalized Pixel Change Rate)
def calculate_npcr(image1, image2):
    # Convert images to NumPy arrays
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)

    # Calculate the number of pixels that differ between the two images
    different_pixels = np.sum(pixels1 != pixels2)

    # Calculate the total number of pixels in the images
    total_pixels = pixels1.size

    # Calculate NPCR
    npcr = different_pixels / total_pixels
    return npcr


# Function to calculate UACI (Unified Average Change Intensity)
def calculate_uaci(image1, image2):
    # Convert images to NumPy arrays
    pixels1 = np.array(image1)
    pixels2 = np.array(image2)

    # Calculate the absolute difference between corresponding pixels in the two images
    abs_difference = np.abs(pixels1.astype(int) - pixels2.astype(int))

    # Calculate the total intensity change
    total_intensity_change = np.sum(abs_difference)

    # Calculate the maximum possible intensity change (255 * number of pixels)
    max_intensity_change = 255 * pixels1.size

    # Calculate UACI
    uaci = total_intensity_change / max_intensity_change
    return uaci

# A function to combine th input, encrypted and decrypted image
def combine_images_with_labels(image1, image2, image3, labels):
    # Get the dimensions of the input images
    width1, height1 = image1.size
    width2, height2 = image2.size
    width3, height3 = image3.size

    # Calculate the total width and height for the combined image
    total_width = width1 + width2 + width3
    max_height = max(height1, height2, height3)

    # Create a new blank image with the combined dimensions
    combined_image = Image.new('RGB', (total_width, max_height), color='white')

    # Paste the input images onto the combined image
    combined_image.paste(image1, (0, 0))
    combined_image.paste(image2, (width1, 0))
    combined_image.paste(image3, (width1 + width2, 0))

    # Create a new image for labels
    label_image = Image.new('RGB', (total_width, 30), color='white')
    draw = ImageDraw.Draw(label_image)
    font = ImageFont.truetype("arial.ttf", 16)  # Adjust font size and type as needed

    x_offset = 0
    for label, width in zip(labels, [width1, width2, width3]):
        label_size = (len(label) * 10, 16)  # Estimate label size based on text length and font size
        label_position = (x_offset + (width - label_size[0]) // 2, 5)
        draw.text(label_position, label, font=font, fill='black')
        x_offset += width

    # Concatenate the label image with the combined image vertically
    combined_image_with_labels = Image.new('RGB', (total_width, max_height + 30), color='white')
    combined_image_with_labels.paste(label_image, (0, max_height))
    combined_image_with_labels.paste(combined_image, (0, 0))

    return combined_image_with_labels


# A variable to store the input image path
input_image_path = "input_image.jpg"

# Generate a random IV (Initialization Vector)
iv = os.urandom(16)

# Apply encryption techniques

# Step 1: Logistic Map Encryption
encrypted_image = encrypt_logistic_map(input_image_path)


# Step 2: Stream Cipher Encryption
# A variable to store a generated for stream cipher encryption and decryption
encryption_key_stream = generate_key()  
encrypted_image_stream = apply_stream_cipher(encrypted_image, encryption_key_stream)

# Step 3: Block Cipher Encryption
# A global variable to store the encrypted data bytes which can be later on directly passed to block cipher decryption
encrypted_data_to_pass=None

# A variable to store block cipher's IV
block_cipher_iv = iv

# A variable to store a generated for stream cipher encryption and decryption
encryption_key_block = generate_key()
encrypted_image_block = apply_block_cipher(encrypted_image_stream, encryption_key_block, block_cipher_iv)


# Save encrypted image
encrypted_image_block.save("encrypted_image.jpg")

# Apply decryption techniques
# Step 1: Block Cipher Decryption
decrypted_image_block = apply_block_cipher_decryption(encrypted_data_to_pass, encryption_key_block, block_cipher_iv)

# Step 2: Stream Cipher Decryption
decrypted_bytes_stream = apply_stream_cipher_decryption(decrypted_image_block, encryption_key_stream)

# Converting the decrypted bytes obtained from stream cipher decryption to an image to be passed into logistic map decryption.
decrypted_image_stream = Image.frombytes(encrypted_image_block.mode, encrypted_image_block.size, decrypted_bytes_stream)

# Step 3: Logistic Map Decryption
decrypted_image = decrypt_logistic_map(decrypted_image_stream)

# Save decrypted image
decrypted_image.save("decrypted_image.jpg")

# Saving decrypted image separately to be used in npcr anad uaci, as if not its getting modified in combined image process.
decrypted_image_1 = decrypted_image

# Open and load the input, encrypted, and decrypted images, those imgs are already saved, so gotta be as opening.
input_image = Image.open("input_image.jpg")
encrypted_image = Image.open("encrypted_image.jpg")
decrypted_image = Image.open("decrypted_image.jpg")

# Define labels for each image
labels = ["Input Image", "Encrypted Image", "Decrypted Image"]

# Combine the images with labels
combined_image_with_labels = combine_images_with_labels(input_image, encrypted_image, decrypted_image, labels)

# Display the combined image with labels
combined_image_with_labels.show()

# Save combined image
combined_image_with_labels.save("Combined_Image.jpg")

# Calculate NPCR and UACI for the entire encryption
npcr = calculate_npcr(input_image, encrypted_image_block)
uaci = calculate_uaci(input_image, encrypted_image_block)
print("NPCR (Entire Encryption) in %:", npcr*100, "%")
print("UACI (Entire Encryption) in %:", uaci*100, "%")

# Calculate NPCR and UACI for the entire decryption
npcr = calculate_npcr(input_image, decrypted_image_1)
uaci = calculate_uaci(input_image, decrypted_image_1)
print("NPCR (Entire Decryption) in %:", npcr*100, "%")
print("UACI (Entire Decryption) in %:", uaci*100, "%")

