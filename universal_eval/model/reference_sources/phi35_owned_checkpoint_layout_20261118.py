"""Names only: own Phi-3.5 checkpoint to the installed HF5 CLIP namespace.

HF5 flattens CLIPVisionModel. The original Phi3 image wrapper also references
the original text embedding; that adds one meta alias, not an extra weight.
"""


def source_to_hf5_meta_key(name):
    old = 'model.vision_embed_tokens.img_processor.vision_model.'
    new = 'model.vision_embed_tokens.img_processor.'
    return new + name[len(old):] if name.startswith(old) else name


def validate_own_meta_mapping(physical_headers, meta_shapes):
    assert len(physical_headers) == 592 and len(meta_shapes) == 593
    mapping = {key: source_to_hf5_meta_key(key) for key in physical_headers}
    assert len(set(mapping.values())) == 592
    assert set(meta_shapes) - set(mapping.values()) == {'model.vision_embed_tokens.wte.weight'}
    assert set(mapping.values()) - set(meta_shapes) == set()
    for key, target in mapping.items():
        assert physical_headers[key]['dtype'] == 'BF16'
        assert physical_headers[key]['shape'] == meta_shapes[target]
    return mapping
