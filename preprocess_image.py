from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


def preprocess_image(image_path, target_size=(50, 50)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size, Image.LANCZOS)
    img_gray = img_resized.convert('L')
    img_array = np.array(img_gray)
    img_flat = img_array.flatten()
    return img_gray, img_flat


desktop = os.path.join(os.path.expanduser("~"), "Desktop")

image_name = "image1.jpeg"
image_path = os.path.join(desktop, image_name)


processed_image, flat_array = preprocess_image(image_path)


plt.figure(figsize=(5, 5))
plt.imshow(processed_image, cmap='gray')
plt.title("Processed Image (50x50 pixels)")
plt.axis('off')
plt.show()

print(f"处理后的图像数据形状: {flat_array.shape}")
print("一维数组的前20个元素:")
print(flat_array[:100])


plt.figure(figsize=(10, 3))
plt.plot(flat_array)
plt.title("Image1 of 1D array")
plt.xlabel("Pixel Index")
plt.ylabel("Pixel Value")
plt.show()