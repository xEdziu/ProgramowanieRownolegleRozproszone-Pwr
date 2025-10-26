import socket
from PIL import Image
import numpy as np
from multiprocessing import cpu_count
from common import send_all, receive_all

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
    arrays = [Image.fromarray(f.astype(np.uint8)) for f in trimmed]
    total_height = sum(a.height for a in arrays)
    width = arrays[0].width
    result = Image.new('RGB', (width, total_height))
    y_offset = 0
    for a in arrays:
        result.paste(a, (0, y_offset))
        y_offset += a.height
    return result

def server_main(image_path, n_clients, host='127.0.0.1', port=2040):
    image = Image.open(image_path)
    fragments = split_image(image, n_clients)
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(n_clients)
    print(f"Serwer nasłuchuje na {host}:{port}...")

    processed_fragments = []
    for i in range(n_clients):
        client_socket, client_address = server_socket.accept()
        print(f"Połączono z klientem {i+1}: {client_address}")

        send_all(client_socket, fragments[i])
        processed_fragment = receive_all(client_socket)
        processed_fragments.append(processed_fragment)

        client_socket.close()

    result_image = merge_image(processed_fragments, n_clients)
    result_image.save("distributed_processed.png")
    print("Obraz zapisany jako distributed_processed.png")

if __name__ == "__main__":
    server_main("fantulka.jpg", n_clients=2, host='192.168.66.2')
