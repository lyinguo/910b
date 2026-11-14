"""Recreate the independently checked original PIL pixels and own tokenizer inputs."""

import hashlib
import json
import math
from pathlib import Path
import re

from jinja2 import Environment
import numpy as np
from PIL import Image
from tokenizers import Tokenizer


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_sha(value):
    return hashlib.sha256(memoryview(np.ascontiguousarray(value))).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def original_resize(image, width, height):
    if image.width != width:
        image = image.resize((width, image.height), Image.Resampling.LANCZOS)
    if image.height != height:
        image = image.resize((image.width, height), Image.Resampling.LANCZOS)
    return image


def original_tiles(image):
    ratio = image.width / image.height
    if image.width >= image.height:
        width, height = 2048, int(2048 / ratio)
        height += height % 2
    else:
        width, height = int(2048 * ratio), 2048
        width += width % 2
    width, height = max(width, 1), max(height, 1)
    first = original_resize(image, width, height)
    ratio = width / height
    if width >= height:
        width = math.ceil(width / 512) * 512
        height = math.ceil(int(width / ratio) / 512) * 512
    else:
        height = math.ceil(height / 512) * 512
        width = math.ceil(int(height * ratio) / 512) * 512
    aligned = original_resize(first, width, height)
    rows, cols = height // 512, width // 512
    tiles = [
        aligned.crop((col * 512, row * 512, (col + 1) * 512, (row + 1) * 512))
        for row in range(rows)
        for col in range(cols)
    ]
    tiles.append(original_resize(aligned, 512, 512))
    return rows, cols, tiles


def placeholders(rows, cols):
    tokens = "<image>" * 64
    fake = "<fake_token_around_image>"
    split = [
        "".join(fake + f"<row_{row + 1}_col_{col + 1}>" + tokens for col in range(cols))
        for row in range(rows)
    ]
    return "\n".join(split) + "\n\n" + fake + "<global-img>" + tokens + fake


class SmolDoclingOwnInputs:
    def __init__(self, weights, protocol):
        for name, expected in protocol["resources_sha256"].items():
            assert Path(name).name == name and file_sha(weights / name) == expected, name
        pre = read_json(weights / "preprocessor_config.json")
        assert pre["size"] == {"longest_edge": 2048}
        assert pre["max_image_size"] == {"longest_edge": 512} and pre["resample"] == 1
        assert pre["image_mean"] == pre["image_std"] == [0.5, 0.5, 0.5]
        assert all(pre[k] for k in ("do_resize", "do_rescale", "do_normalize", "do_pad",
                                    "do_convert_rgb", "do_image_splitting"))
        assert read_json(weights / "processor_config.json")["image_seq_len"] == 64
        self.tokenizer = Tokenizer.from_file(str(weights / "tokenizer.json"))
        template = read_json(weights / "chat_template.json")["chat_template"]
        self.template = Environment(trim_blocks=True, lstrip_blocks=True).from_string(template)
        lut = (np.arange(256, dtype=np.float64) * pre["rescale_factor"]).astype(np.float32)
        self.lut = (lut - np.float32(0.5)) / np.float32(0.5)
        assert file_sha(protocol["original_A100_rows"]) == protocol["original_A100_rows_SHA256"]
        assert file_sha(protocol["independent_R_inputs"]) == protocol["independent_R_inputs_SHA256"]
        self.original = [
            row for row in read_jsonl(protocol["original_A100_rows"])
            if row["event"] == "actual_own_HF_sample"
        ]
        self.independent = read_jsonl(protocol["independent_R_inputs"])
        assert [r["index"] for r in self.original] == list(range(200))
        assert [r["index"] for r in self.independent] == list(range(200))

    def prepare(self, index, prompt, media_path):
        own, ref = self.original[index], self.independent[index]
        assert prompt == own["candidate_exact_archived_prompt"], (index, "prompt")
        media = Path(media_path)
        assert media.resolve() == Path(ref["R_media"]).resolve(), (index, "media path")
        assert file_sha(media) == ref["original_media_SHA256"] == own["own_media_SHA256"]
        with Image.open(media) as source:
            image = source.convert("RGB")
        assert array_sha(np.asarray(image)) == own["RGB_SHA256"]
        rows, cols, tiles = original_tiles(image)
        assert [rows, cols] == ref["pixel_grid"] and len(tiles) == ref["image_tile_count"]
        pixels = np.stack([self.lut[np.asarray(tile)].transpose(2, 0, 1) for tile in tiles])[None]
        chat = self.template.render(
            messages=[{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": prompt},
            ]}],
            add_generation_prompt=True,
        )
        assert chat == own["rendered_prompt"] and chat.count("<image>") == 1
        expanded = chat.replace("<image>", placeholders(rows, cols))
        ids = self.tokenizer.encode(expanded, add_special_tokens=True).ids
        assert ids == own["input_ids"] == ref["input_ids"]
        assert ids.count(49190) == 64 * len(tiles)
        ids = np.asarray([ids], dtype=np.int64)
        arrays = {
            "input_ids": ids,
            "attention_mask": np.ones_like(ids),
            "pixel_values": pixels,
            "pixel_attention_mask": np.ones((1, len(tiles), 512, 512), dtype=np.int64),
        }
        for name, value in arrays.items():
            assert {
                "shape": list(value.shape),
                "dtype": "torch." + str(value.dtype),
                "SHA256": array_sha(value),
            } == own["native_inputs"][name], (index, name)
        return arrays, ref

    def decode(self, tokens):
        raw = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return re.sub(r"(\n?<global-img>\n?|<row_\d+_col_\d+>\n?)+", "", raw).strip()
