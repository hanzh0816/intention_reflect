#!/usr/bin/env python3
"""Test script to check msgpack.xz file format."""

import lzma
import msgpack
import pickle

file_path = 'work_dirs/scenario_sim/0cccf163_closed_loop_reactive_agents/simulation_logs/ImitationPlanner/traversing_pickup_dropoff/2021.07.22.16.04.21_veh-35_02539_05454/0cccf1639991539a/0cccf1639991539a.msgpack.xz'

print(f"Loading: {file_path}")

with open(file_path, 'rb') as f:
    compressed = f.read()

print(f"Compressed size: {len(compressed)} bytes")

decompressed = lzma.decompress(compressed)
print(f"Decompressed size: {len(decompressed)} bytes")
print(f"First 20 bytes (hex): {decompressed[:20].hex()}")

# Try msgpack
try:
    data = msgpack.unpackb(decompressed, raw=False, strict_map_key=False)
    print('✓ MessagePack unpacking successful!')
    print(f'  Type: {type(data)}')

    if isinstance(data, bytes):
        print(f'  Data is bytes, size: {len(data)}')
        print(f'  First 20 bytes: {data[:20].hex()}')

        # Try pickle on the bytes
        obj = pickle.loads(data)
        print('✓ Pickle unpacking successful!')
        print(f'  Final type: {type(obj)}')
        print(f'  Has simulation_history: {hasattr(obj, "simulation_history")}')

        if hasattr(obj, 'simulation_history'):
            history = obj.simulation_history
            print(f'  History type: {type(history)}')
            print(f'  History samples: {len(history.data) if hasattr(history, "data") else "N/A"}')

except Exception as e:
    print(f'✗ Error: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
