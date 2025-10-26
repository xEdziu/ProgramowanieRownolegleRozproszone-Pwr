import socket
import numpy as np
from PIL import Image
from common import send_all, receive_all

def sobel_filter(image_fragment: np.ndarray) -> np.ndarray:
    Kx = np.array([[1, 0, -1],
                   [2, 0, -2],
                   [1, 0, -1]])
    Ky = np.array([[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]])

    gray = np.mean(image_fragment, axis=2)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)

    for i in range(1, gray.shape[0]-1):
        for j in range(1, gray.shape[1]-1):
            region = gray[i-1:i+2, j-1:j+2]
            gx[i, j] = np.sum(Kx * region)
            gy[i, j] = np.sum(Ky * region)

    g = np.sqrt(gx**2 + gy**2)
    g = np.clip(g, 0, 255)
    return np.stack((g,)*3, axis=-1).astype(np.uint8)

def client_main(host='127.0.0.1', port=2040):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    print("Połączono z serwerem")

    fragment = receive_all(client_socket)
    print("Otrzymano fragment")
    processed_fragment = sobel_filter(fragment)
    send_all(client_socket, processed_fragment)

    client_socket.close()
    print("Fragment przetworzony i wysłany")

if __name__ == "__main__":
    client_main(host="192.168.66.2")
