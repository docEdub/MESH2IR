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
import websockets
import json

async def process_go_message(positions):
    # Replace with your own logic to process positions
    print("Received positions:", positions)

async def handle_websocket(websocket, path):
    async for message in websocket:
        try:
            data = json.loads(message)
            command = data.get("command")
            positions = data.get("positions")
            
            if command == "go" and isinstance(positions, list) and len(positions) == 2:
                await process_go_message(positions)
            else:
                print("Invalid message or command received")
        except json.JSONDecodeError:
            print("Message is not valid JSON")

start_server = websockets.serve(handle_websocket, "0.0.0.0", 6789)

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
