import numpy as np
from scipy.integrate import quad
from PIL import Image
import os


def preprocess_image(image_path, target_size=(50, 50)):
    img = Image.open(image_path)
    img_resized = img.resize(target_size, Image.LANCZOS)
    img_gray = img_resized.convert('L')
    img_array = np.array(img_gray)
    img_flat = img_array.flatten()
    return img_gray, img_flat



desktop = os.path.join(os.path.expanduser("~"), "Desktop")

image_names = ["image1.jpeg", "image2.jpeg", "image3.jpeg", "image4.jpeg", "image5.jpeg", "image6.jpeg"]
image_paths = [os.path.join(desktop, name) for name in image_names]


flat_arrays = []

for path in image_paths:
    _, flat_array = preprocess_image(path)
    flat_arrays.append(flat_array)


def kernel_function(u):
    return np.exp(-u ** 2 / 2) / np.sqrt(2 * np.pi)


def survival_function(x, data, bandwidth):
    n = len(data)
    return (1 / n) * np.sum([1 - kernel_function((x - value) / bandwidth) for value in data])



def integrand(x, X_data, Y_data, bandwidth):
    F_hat = survival_function(x, X_data, bandwidth)
    G_hat = survival_function(x, Y_data, bandwidth)

    return 1 / x ** (2) * (F_hat) ** 5 * (G_hat) ** 5


def calculate_D(X_data, Y_data, bandwidth):
    result, _ = quad(integrand, 0, np.inf, args=(X_data, Y_data, bandwidth))
    return result


bandwidths = [50, 70, 90]

for bandwidth in bandwidths:
    print(f"当前带宽为: {bandwidth}")
    results_dict = {}
    for i in range(len(flat_arrays)):
        for j in range(len(flat_arrays)):
            X_data = flat_arrays[i]
            Y_data = flat_arrays[j]
            D_F_G = calculate_D(X_data, Y_data, bandwidth)
            key = f"Image_{i + 1}_vs_Image_{j + 1}"
            results_dict[key] = D_F_G

    results_dict = {}

    for i in range(len(flat_arrays)):
        for j in range(len(flat_arrays)):
            X_data = flat_arrays[i]
            Y_data = flat_arrays[j]
            D_F_G = calculate_D(X_data, Y_data, bandwidth)
            key = f"Image_{i + 1}_vs_Image_{j + 1}"
            results_dict[key] = D_F_G

    latex_table = "\\begin{tabular}{|c|" + "c|c|c|c|c|c|}\n\\hline\n"
    latex_table += " & " + " & ".join(image_names) + " \\\\ \\hline\n"
    for i in range(len(image_names)):
        latex_table += image_names[i] + " "
        for j in range(len(image_names)):
            key = f"Image_{i + 1}_vs_Image_{j + 1}"
            value = results_dict.get(key, 0)
            latex_table += f"& {value:.6f}"
        latex_table += " \\\\ \\hline\n"
    latex_table += "\\end{tabular}"

    print(latex_table)
