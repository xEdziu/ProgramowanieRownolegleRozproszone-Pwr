from multiprocessing import Pool
from PIL import Image
import numpy as np

def sobel_filter(image_fragment: np.ndarray) -> np.ndarray:
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    gray = np.mean(image_fragment, axis=2)  # RGB → grayscale
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            region = gray[i-1:i+2, j-1:j+2]
            gx[i, j] = np.sum(Kx * region)
            gy[i, j] = np.sum(Ky * region)

    g = np.sqrt(gx**2 + gy**2)
    g = np.clip(g, 0, 255)
    return np.stack((g,)*3, axis=-1).astype(np.uint8)  # grayscale → RGB

def process_fragment(fragment):
    return sobel_filter(fragment)

def split_image(image, n_parts, overlap=1):
    width, height = image.size
    part_height = height // n_parts
    fragments = []

    img_array = np.array(image)

    for i in range(n_parts):
        top = i * part_height
        bottom = (i + 1) * part_height if i < n_parts - 1 else height

        top_extended = max(0, top - overlap)
        bottom_extended = min(height, bottom + overlap)

        fragment = img_array[top_extended:bottom_extended, :, :]
        fragments.append(fragment)
    return fragments

def merge_image(fragments, n_parts, overlap=1):
    trimmed = []

    for i, fragment in enumerate(fragments):
        if i == 0:
            trimmed.append(fragment[:-overlap])
        elif i == n_parts - 1:
            trimmed.append(fragment[overlap:])
        else:
            trimmed.append(fragment[overlap:-overlap])

    arrays = [Image.fromarray(f) for f in trimmed]
    total_height = sum(a.height for a in arrays)
    width = arrays[0].width
    result = Image.new('RGB', (width, total_height))
    y_offset = 0
    for a in arrays:
        result.paste(a, (0, y_offset))
        y_offset += a.height
    return result

def main():
    image_path = "fantulka.jpg"
    fileName = image_path.split(".")[0]
    image = Image.open(image_path)
    fragmentsNumber = 4

    fragments = split_image(image, fragmentsNumber)

    with Pool(fragmentsNumber) as p:
        processed_fragments = p.map(process_fragment, fragments)

    result_image = merge_image(processed_fragments, fragmentsNumber)
    result_image.save(f"{fileName}_processed.png")
    print(f"Obraz przetworzony i zapisany jako {fileName}_processed.png")

if __name__ == "__main__":
    main()