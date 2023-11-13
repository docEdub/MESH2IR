import asyncio
import json
import numpy
import struct
import torch
import torch_geometric
import websockets

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch.autograd import Variable

netG_path = "evaluate/Models/MESH2IR-D-EDR/netG_epoch_483.pth"
mesh_net_path = "evaluate/Models/MESH2IR-D-EDR/mesh_net_epoch_483.pth"
batch_size = 128
sampe_rate = 16000
gpus = [0]

ws = None
mesh_embedding = None


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_network_stageI(netG_path,mesh_net_path):
    from model import STAGE1_G, STAGE1_D, MESH_NET
    netG = STAGE1_G()
    netG.apply(weights_init)

    print(netG)

    mesh_net = MESH_NET()

    if netG_path != '':
        state_dict = torch.load(netG_path, map_location = lambda storage, loc: storage)
        netG.load_state_dict(state_dict)
        print('Load from: ', netG_path)

    if mesh_net_path != '':
        state_dict = torch.load(mesh_net_path, map_location=lambda storage, loc: storage)
        mesh_net.load_state_dict(state_dict)
        print('Load from: ', mesh_net_path)

    netG.cuda()
    mesh_net.cuda()
    return netG, mesh_net


# Load AI models.
print("Loading models ...")
netG, mesh_net = load_network_stageI(netG_path,mesh_net_path)
netG.eval()
mesh_net.eval()
netG.to(device='cuda')
mesh_net.to(device='cuda')
print("Loading models - done")


async def on_request_rir_message(positions):
    print("Processing RIR request: positions = ", positions, " ...")

    if mesh_embedding is None:
        print("Room mesh not set, yet. No RIR generated.");
        print("Processing RIR request - done")
        return

    # Expand position data to batch size, if needed.
    positions_count = len(positions)
    last_index = positions_count - 1
    filler = batch_size - positions_count
    if filler < batch_size:
        for i in range(filler):
            positions.append(positions[last_index])

    # Do inferencing.
    positions_embedding = Variable(torch.from_numpy(numpy.array(positions, dtype = numpy.float32))).cuda()

    inputs = (positions_embedding, mesh_embedding)
    lr_fake, fake, _ = torch.nn.parallel.data_parallel(netG, inputs, gpus)

    # Collect output RIRs into one long array, 3968 samples per RIR.
    rirs = []
    for i in range(positions_count):
        fake_IR = numpy.array(fake[i].to("cpu").detach(), dtype=numpy.float32)
        fake_IR_only = fake_IR[:,0:(4096-128)]
        fake_energy = numpy.median(fake_IR[:,(4096 - 128):4096]) * 10
        rir = fake_IR_only * fake_energy
        rirs += rir[0].tolist()

    # Send the RIRs to the websocket client.
    # rirs = list(map(float, rirs))
    # for i in range(len(rirs)):
    #     rirs[i] = float(rirs[i])
    binary_data = struct.pack(f'{len(rirs)}f', *rirs)
    await ws.send(binary_data)

    print("Processing RIR request - done")

async def on_set_room_mesh_message(obj_string):
    print("Processing room mesh ...")

    # Parse .obj data string.
    vertices = []
    faces = []
    for line in obj_string.splitlines():
        parts = line.split()
        if len(parts) > 0:
            if parts[0] == 'v':  # Vertex information.
                vertices.append([float(v) for v in parts[1:4]])
            elif parts[0] == 'f':  # Face information.
                face = [int(idx.split('/')[0]) for idx in parts[1:]]
                faces.append(face)

    # Convert to zero-based indexing.
    faces = [[idx - 1 for idx in face] for face in faces]

    # Create tensors.
    vertices_tensor = torch.tensor(vertices, dtype = torch.float)
    faces_tensor = torch.tensor(faces, dtype = torch.long).t().contiguous()

    # Create torch_geometric data/mesh.
    mesh = Data(pos=vertices_tensor, face=faces_tensor)

    # Create pre-transformed mesh graph torch data.
    pre_transform = torch_geometric.transforms.FaceToEdge()
    mesh_graph = pre_transform(mesh)

    # Create mesh embedding.
    data_list = [mesh_graph] * batch_size
    loader = DataLoader(data_list, batch_size = batch_size)
    data = next(iter(loader))
    data['edge_index'] = Variable(data['edge_index'])
    data['pos'] = Variable(data['pos'])
    data = data.cuda()

    global mesh_embedding
    mesh_embedding = torch.nn.parallel.data_parallel(mesh_net, data, [gpus[0]])

    print("Processing room mesh - done")


async def handle_websocket(websocket, path):
    global ws
    ws = websocket

    async for message in websocket:
        try:
            data = json.loads(message)
            command = data.get("command")
            
            # positions = Array of arrays of 6 floats. First 3 are source position, second 3 are listener position.
            if command == "request-rir":
                positions = data.get("positions")
                if isinstance(positions, list):
                    await on_request_rir_message(positions)

            # obj = String containing .obj data.
            elif command == "set-room-mesh":
                obj_string = data.get("obj")
                if isinstance(obj_string, str):
                    await on_set_room_mesh_message(obj_string)

            else:
                print("Invalid message or command received")

        except json.JSONDecodeError:
            print("Message is not valid JSON")

start_server = websockets.serve(handle_websocket, "0.0.0.0", 6789)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
