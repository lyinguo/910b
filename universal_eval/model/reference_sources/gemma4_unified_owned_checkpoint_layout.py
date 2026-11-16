"""Unweighted own-checkpoint name mapping and independent raw-patch replay.

This does not import the legacy adapter's mock/stub initializer, load learned
weights, or change model math. The source/target shape and coverage guards are
mandatory before these name mappings may be used by a real model loader.
"""
def native_source_key(name):
    if name.startswith('model.vision_embedder.'):
        return 'model.embed_vision.'+name[len('model.vision_embedder.'):]
    if name=='model.embed_vision.embedding_projection.weight':
        return 'model.embed_vision.multimodal_embedder.embedding_projection.weight'
    return name

def owned_mapping(headers,native_shapes):
    mapped={}
    for source,row in headers.items():
        target=native_source_key(source)
        assert target in native_shapes and target not in mapped,(source,target)
        assert list(row['shape'])==list(native_shapes[target]),(source,target,'Own shape differs')
        mapped[target]=source
    assert set(native_shapes)-set(mapped)=={'lm_head.weight'}
    embedding='model.language_model.embed_tokens.weight'
    assert embedding in mapped and list(native_shapes[embedding])==list(native_shapes['lm_head.weight'])
    return mapped,dict(target='lm_head.weight',own_source=mapped[embedding],own_native_embedding=embedding,not_another_checkpoint=True)

def independent_merged_patches(image,max_soft_tokens=280,patch_size=16,pooling_kernel_size=3):
    import math,torch
    from torchvision.transforms.v2 import functional as VF
    from torchvision.transforms import InterpolationMode
    h,w=image.height,image.width;max_patches=max_soft_tokens*pooling_kernel_size**2;multiple=patch_size*pooling_kernel_size
    scale=math.sqrt(max_patches*patch_size**2/(h*w));th=math.floor(h*scale/multiple)*multiple;tw=math.floor(w*scale/multiple)*multiple
    assert th or tw
    if not th:th=multiple;tw=min(math.floor(w/h)*multiple,max_soft_tokens*multiple)
    elif not tw:tw=multiple;th=min(math.floor(h/w)*multiple,max_soft_tokens*multiple)
    assert th*tw<=max_patches*patch_size**2 and th%multiple==tw%multiple==0
    pixels=VF.pil_to_tensor(image)
    if (th,tw)!=(h,w):pixels=VF.resize(pixels,[th,tw],interpolation=InterpolationMode.BICUBIC,antialias=True)
    pixels=pixels.to(torch.float32).mul(1/255)
    # Direct 48px patches independently replay the original 16px patch,
    # spatial sort, 3x3 merge and pixel-axis permutation without calling it.
    nh,nw=th//multiple,tw//multiple;patch=pixels.reshape(3,nh,multiple,nw,multiple).permute(1,3,2,4,0).reshape(nh*nw,multiple*multiple*3)
    pos=torch.stack(torch.meshgrid(torch.arange(nw),torch.arange(nh),indexing='xy'),dim=-1).reshape(nh*nw,2)
    patch=torch.nn.functional.pad(patch,(0,0,0,max_soft_tokens-nh*nw));pos=torch.nn.functional.pad(pos,(0,0,0,max_soft_tokens-nh*nw),value=-1)
    return patch[None],pos[None],dict(original_image_shape=[h,w],resized_shape=[th,tw],valid_merged_patches=nh*nw)
