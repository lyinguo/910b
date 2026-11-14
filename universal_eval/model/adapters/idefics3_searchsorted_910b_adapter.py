"""Instance-local CANN searchsorted compatibility; original BF16 coordinates are retained."""
import types
import torch
from universal_eval.model.adapters.idefics3_native_910b_adapter import Idefics3Native910BAdapter

def npu_compatible_vision_embeddings(self,pixel_values,patch_attention_mask):
    batch_size,_,height,width=pixel_values.shape
    embeddings=self.patch_embedding(pixel_values).flatten(2).transpose(1,2)
    patches_h,patches_w=height//self.patch_size,width//self.patch_size
    boundaries=torch.arange(1/self.num_patches_per_side,1.0,1/self.num_patches_per_side,device=pixel_values.device)
    position_ids=torch.full((batch_size,patches_h*patches_w),fill_value=0,device=pixel_values.device)
    n_h=patch_attention_mask[:,:,0].sum(1);n_w=patch_attention_mask[:,0,:].sum(1)
    step_h=1.0/n_h;step_w=1.0/n_w
    h_indices=torch.arange(patch_attention_mask.size(1),device=pixel_values.device,dtype=torch.float32)
    w_indices=torch.arange(patch_attention_mask.size(2),device=pixel_values.device,dtype=torch.float32)
    coords_h=(h_indices[None,:]*step_h[:,None]).clamp(max=1.0-1e-6).to(pixel_values.dtype)
    coords_w=(w_indices[None,:]*step_w[:,None]).clamp(max=1.0-1e-6).to(pixel_values.dtype)
    # CANN SearchSorted rejects BF16. FP32 represents every BF16 value exactly; no coordinate is recomputed or rounded differently.
    bucket_h=torch.bucketize(coords_h.float(),boundaries,right=True)
    bucket_w=torch.bucketize(coords_w.float(),boundaries,right=True)
    values=(bucket_h[:,:,None]*self.num_patches_per_side+bucket_w[:,None,:]).reshape(batch_size,-1)
    mask=patch_attention_mask.view(batch_size,-1);position_ids[mask]=values[mask]
    return embeddings+self.position_embedding(position_ids)

class Idefics3Searchsorted910BAdapter(Idefics3Native910BAdapter):
    def load(self,model_path,**kwargs):
        super().load(model_path,**kwargs)
        embeddings=self.model.model.vision_model.embeddings
        embeddings.forward=types.MethodType(npu_compatible_vision_embeddings,embeddings)
        self.lineage['NPU_compatibility']={'scope':'Only this model instance vision embedding forward',
            'reason':'CANN SearchSorted has no BF16 kernel input; represent the already BF16-rounded coordinates exactly in FP32 for searchsorted only.',
            'learned_computation':'Patch convolution, positional embedding lookup, vision transformer, connector and language model remain actual NPU BF16 computation with all original weights.',
            'no_shared_source_changes':True,'no_CUDA_or_global_framework_shims':True}
