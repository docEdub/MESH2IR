'''
Steps:
- Start websocket server.
- On simplified room mesh received ...
    - Convert it to torch geometry mesh:
        - https://chat.openai.com/c/81937c74-ff79-42c9-9ea9-0cb87ba56f0d
    - Pre-transform the torch geometry mesh.
        - Search graph_generatory.py for `pre_transform` for how.
    - Keep the pre-transformed mesh in memory while waiting for listener and sound source coords to be received.
'''

import asyncio
import json
import torch
import torch_geometric
import websockets

from torch_geometric.data import Data

room_mesh_graph = {}
listener_position = [ 0.0, 0.0, 0.0 ]
sound_positions = [ [ 0.0, 0.0, 0.0 ] ]

async def process_go_message(positions):
    # Replace with your own logic to process positions
    print("Received positions:", positions)

async def process_listener_position_message(position):
    print("Listener position:", position)
    listener_position = position

async def process_room_mesh_message(obj_string):
    # Parse .obj data string.
    vertices = []
    faces = []
    for line in obj_string.splitlines():
        parts = line.split()
        if len(parts) > 0:
            if parts[0] == 'v':  # Vertex information
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'f':  # Face information
                face = [int(idx.split('/')[0]) for idx in parts[1:]]
                faces.append(face)

    # Convert to zero-based indexing
    faces = [[idx - 1 for idx in face] for face in faces]

    # Create tensors
    vertices_tensor = torch.tensor(vertices, dtype=torch.float)
    faces_tensor = torch.tensor(faces, dtype=torch.long).t().contiguous()

    # Create torch_geometric data/mesh
    mesh = Data(pos=vertices_tensor, face=faces_tensor)

    # Create pre-transformed "pickle" data.
    pre_transform = torch_geometric.transforms.FaceToEdge()
    room_mesh_graph = pre_transform(mesh)

    print("Created room mesh torch graph")


async def handle_websocket(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            command = data.get("command")
            
            if command == "go":
                positions = data.get("positions")
                if isinstance(positions, list) and len(positions) == 2:
                    await process_go_message(positions)
            elif command == "listener-position":
                position = data.get("position")
                if isinstance(position, list) and len(position) == 3:
                    await process_listener_position_message(position)
            elif command == "room-mesh":
                obj_string = data.get("obj")
                if isinstance(obj_string, str):
                    await process_room_mesh_message(obj_string)
            else:
                print("Invalid message or command received")
        except json.JSONDecodeError:
            print("Message is not valid JSON")

start_server = websockets.serve(handle_websocket, "0.0.0.0", 6789)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
