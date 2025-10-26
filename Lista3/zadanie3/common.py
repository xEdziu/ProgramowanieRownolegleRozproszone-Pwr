import json
import numpy as np
import struct

def send_all(sock, data):
    # Convert numpy array to a list for JSON serialization
    serialized_data = json.dumps(data.tolist()).encode('utf-8')
    # Send the size of the data first (4 bytes, big-endian)
    sock.sendall(struct.pack('>I', len(serialized_data)))
    # Send the actual serialized data
    sock.sendall(serialized_data)

def receive_all(sock):
    # Receive the size of the incoming data (4 bytes, big-endian)
    data_size = struct.unpack('>I', sock.recv(4))[0]
    buffer = b""
    while len(buffer) < data_size:
        part = sock.recv(4096)
        if not part:
            break
        buffer += part
    # Deserialize JSON and convert back to numpy array
    data = np.array(json.loads(buffer.decode('utf-8')))
    return data