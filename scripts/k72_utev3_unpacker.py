#!/usr/bin/env python3
"""K72 / GloryFitPro UTE-Tech V3 watch-face unpacker.

Extracts the image resources from a UTE-Tech V3 .bin face using raw LZ4 blocks.
Tested against the supplied K72 "Double Circle.bin" (410x502).

Usage:
    python k72_utev3_unpacker.py "Double Circle.bin"
    python k72_utev3_unpacker.py face.bin -o extracted

Requires:
    pip install lz4 pillow
"""
from __future__ import annotations
import argparse, json, struct
from pathlib import Path
import lz4.block
from PIL import Image

HEADER_SIZE = 0x18
ITEM_SIZE = 0x18
FRAME_HEADER_SIZE = 20
ITEM_COUNT = 14


def u16(b: bytes, o: int) -> int:
    return struct.unpack_from('<H', b, o)[0]


def u32(b: bytes, o: int) -> int:
    return struct.unpack_from('<I', b, o)[0]


def rgb565_to_rgb(raw: bytes) -> bytes:
    out = bytearray((len(raw) // 2) * 3)
    j = 0
    for i in range(0, len(raw), 2):
        v = raw[i] | (raw[i + 1] << 8)
        r = ((v >> 11) & 0x1F) * 255 // 31
        g = ((v >> 5) & 0x3F) * 255 // 63
        b = (v & 0x1F) * 255 // 31
        out[j:j + 3] = bytes((r, g, b))
        j += 3
    return bytes(out)


def decode_frame(blob: bytes, offset: int):
    if offset + FRAME_HEADER_SIZE > len(blob):
        raise ValueError(f'Frame header at 0x{offset:X} is truncated')

    resource_offset = u32(blob, offset)
    width = u16(blob, offset + 4)
    height = u16(blob, offset + 6)
    pixel_format = u32(blob, offset + 8)
    compressed_size = u32(blob, offset + 12)
    flags = blob[offset + 16:offset + 20]

    payload_start = offset + FRAME_HEADER_SIZE
    payload_end = payload_start + compressed_size
    if payload_end > len(blob):
        raise ValueError(f'Frame payload at 0x{offset:X} runs past EOF')

    if pixel_format == 0x200:
        bpp = 2
    elif pixel_format == 0x301:
        bpp = 3
    else:
        raise ValueError(f'Unknown pixel format 0x{pixel_format:08X} at 0x{offset:X}')

    raw_size = width * height * bpp
    compressed = blob[payload_start:payload_end]
    raw = lz4.block.decompress(compressed, uncompressed_size=raw_size)
    if len(raw) != raw_size:
        raise ValueError(f'LZ4 size mismatch: got {len(raw)}, expected {raw_size}')

    if pixel_format == 0x200:
        rgb = rgb565_to_rgb(raw)
    else:
        # UTE V3 RGB888 is stored as R,G,B bytes.
        rgb = raw

    image = Image.frombytes('RGB', (width, height), rgb)
    meta = {
        'offset': offset,
        'resource_offset': resource_offset,
        'width': width,
        'height': height,
        'pixel_format': f'0x{pixel_format:08X}',
        'compressed_size': compressed_size,
        'decompressed_size': raw_size,
        'flags': flags.hex(' '),
        'next_offset': payload_end,
    }
    return image, payload_end, meta


def main() -> None:
    parser = argparse.ArgumentParser(description='Unpack GloryFitPro UTE-Tech V3 watch-face BIN files')
    parser.add_argument('input', type=Path)
    parser.add_argument('-o', '--output', type=Path)
    parser.add_argument('--no-dedupe', action='store_true', help='extract shared resources again for each item')
    args = parser.parse_args()

    source = args.input
    blob = source.read_bytes()
    if len(blob) < HEADER_SIZE + ITEM_COUNT * ITEM_SIZE:
        raise SystemExit('File is too small to be a K72 UTE-Tech V3 face')

    output = args.output or source.with_name(source.stem + '_unpacked')
    output.mkdir(parents=True, exist_ok=True)

    canvas_w, canvas_h = u16(blob, 0x0C), u16(blob, 0x0E)
    body_length = u32(blob, 0x04)

    manifest = {
        'source': source.name,
        'file_size': len(blob),
        'body_length': body_length,
        'canvas': [canvas_w, canvas_h],
        'format': 'UTE-Tech V3 / raw LZ4 blocks',
        'items': [],
    }

    # Several descriptors can point to the same resource stream. Cache decoded streams
    # so the extracted data is not unnecessarily duplicated.
    cache = {}

    print(f'Input : {source}')
    print(f'Size  : {len(blob):,} bytes')
    print(f'Canvas: {canvas_w} x {canvas_h}')
    print()

    for item_index in range(ITEM_COUNT):
        base = HEADER_SIZE + item_index * ITEM_SIZE
        item_type = u16(blob, base)
        item_w = u16(blob, base + 2)
        data_offset = u32(blob, base + 4)
        item_h = u16(blob, base + 8)
        x = u16(blob, base + 10)
        y = u16(blob, base + 12)
        frame_count = u16(blob, base + 16)

        item_dir = output / f'item_{item_index:02d}_type_{item_type:02d}'
        item_dir.mkdir(exist_ok=True)
        item_manifest = {
            'index': item_index,
            'type': item_type,
            'size': [item_w, item_h],
            'position': [x, y],
            'frame_count': frame_count,
            'data_offset': data_offset,
            'frames': [],
        }

        print(f'Item {item_index:02d}: type={item_type:2d} size={item_w}x{item_h} '
              f'pos=({x},{y}) frames={frame_count} data=0x{data_offset:X}')

        if data_offset in cache and not args.no_dedupe:
            cached = cache[data_offset]
            for frame_no, (image, meta) in enumerate(cached):
                filename = item_dir / f'frame_{frame_no:03d}.png'
                image.save(filename)
                item_manifest['frames'].append({**meta, 'file': filename.name, 'shared_resource': True})
            manifest['items'].append(item_manifest)
            print('  shared resource: reused decoded frames')
            continue

        frames = []
        cursor = data_offset
        for frame_no in range(frame_count):
            image, cursor, meta = decode_frame(blob, cursor)
            filename = item_dir / f'frame_{frame_no:03d}.png'
            image.save(filename)
            frames.append((image.copy(), meta.copy()))
            item_manifest['frames'].append({**meta, 'file': filename.name, 'shared_resource': False})
            print(f'  frame {frame_no:03d}: {meta["width"]}x{meta["height"]} '
                  f'{meta["pixel_format"]} LZ4 {meta["compressed_size"]:,} -> '
                  f'{meta["decompressed_size"]:,} bytes')

        cache[data_offset] = frames
        manifest['items'].append(item_manifest)

    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print()
    print(f'Done. Extracted files are in: {output}')


if __name__ == '__main__':
    main()
