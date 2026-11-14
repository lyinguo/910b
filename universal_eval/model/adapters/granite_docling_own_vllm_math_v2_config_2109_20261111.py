"""Own GraniteDocling original A100 vLLM packed graph, Ascend BF16 reference.

This is an unapproved mathematical component, not a model acceptance or a
publication tool. Scheduler, original input protocol and complete verification
are separate. No CUDA paged-cache acceleration or CPU equality is claimed.
"""
from __future__ import annotations
import hashlib,json,math
from pathlib import Path
import torch
from torch.nn import functional as F

def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def native_rms(value,weight,residual=None):
    """Original vllm_c BF16 residual sum, then F32 variance and weight product.

    The default native-IR implementation has a different intermediate cast.
    This original CUDA-kernel expression requires the genuine original-context
    operator control; it must never be selected by a score comparison.
    """
    summed=value.float()
    if residual is not None:summed=(summed+residual.float()).to(value.dtype).float()
    normalized=summed*torch.rsqrt(summed.square().mean(-1,keepdim=True)+1e-5)
    output=(normalized*weight.float()).to(value.dtype)
    return output if residual is None else (output,summed.to(value.dtype))

def native_silu_mul(packed):
    gate,up=packed.chunk(2,-1)
    # Original CUDA silu_and_mul rounds the SiLU intermediate to source BF16.
    # Genuine same-input CUDA control rejects the final-only F32 cast.
    return F.silu(gate)*up

class SourceWeights:
    """All471 source/cast/actual-NPU SHA, complete packed slices and two heads."""
    def __init__(self,root,manifest,device,progress=None):
        from safetensors import safe_open
        assert device.type=='npu'
        self.root,self.device=Path(root).resolve(),device
        self.values={};self.records=[];self.original_files={};expected=manifest['source_tensors']
        assert len(expected)==471 and 'model.safetensors' in manifest['source_files']
        for name,record in manifest['source_files'].items():
            assert Path(name).name==name
            path=self.root/name;assert path.is_file() and file_sha(path)==record['SHA256'],name
            st=path.stat();self.original_files[name]=(st.st_size,st.st_mtime_ns,st.st_ino)
        slices={};visited=set()
        with safe_open(str(self.root/'model.safetensors'),framework='pt',device='cpu') as handle:
            assert set(handle.keys())==set(expected)-{'lm_head.weight'} and len(handle.keys())==470
            for key in sorted(expected):
                physical_source='model.text_model.embed_tokens.weight' if key=='lm_head.weight' else key
                raw=handle.get_tensor(physical_source);record=expected[key]
                assert str(raw.dtype)==record['dtype']=='torch.bfloat16'
                assert list(raw.shape)==record['shape'] and tensor_sha(raw)==record['SHA256'],key
                name,start,total=self.packed_mapping(key,expected)
                if name not in self.values:
                    shape=list(raw.shape)
                    if total is not None:shape[0]=total
                    self.values[name]=torch.empty(shape,device=device,dtype=torch.bfloat16)
                physical=self.values[name];target=physical if start is None else physical[start:start+raw.shape[0]]
                assert target.shape==raw.shape;target.copy_(raw)
                actual_sha=tensor_sha(target)
                assert actual_sha==record['SHA256'] and torch.equal(target.cpu(),raw)
                self.records.append(dict(source=key,source_file='model.safetensors',source_physical_key=physical_source,head_materialized_from_own_embedding=(key=='lm_head.weight'),source_shape=list(raw.shape),source_dtype=str(raw.dtype),source_SHA256=record['SHA256'],cast_SHA256=record['SHA256'],actual_NPU_SHA256=actual_sha,packed_parameter=name,packed_slice=None if start is None else [start,start+raw.shape[0]],physical_shape=list(physical.shape),device=str(device)))
                slices.setdefault(name,[]).append((0 if start is None else start,raw.shape[0],key));visited.add(key)
                if progress and len(visited)%40==0:progress({'source_tensors':len(visited),'expected_source_tensors':471})
        assert visited==set(expected) and len(self.values)==333
        for name,covered in slices.items():
            end=0
            for start,length,key in sorted(covered):
                assert start==end,('Packed overlap or gap',name,key,start,end);end+=length
            assert end==self.values[name].shape[0]
        assert self.values['lm_head.weight'].shape==self.values['model.text_model.embed_tokens.weight'].shape==(100352,576)
        assert self.values['lm_head.weight'].data_ptr()!=self.values['model.text_model.embed_tokens.weight'].data_ptr()
        self.physical=[dict(name=name,shape=list(t.shape),dtype=str(t.dtype),SHA256=tensor_sha(t),device=str(device)) for name,t in sorted(self.values.items())]
        expected_physical={x['name']:x for x in manifest['genuine_A100_physical_parameters']}
        assert set(expected_physical)==set(self.values)
        for row in self.physical:
            aa=expected_physical[row['name']]
            assert row['shape']==aa['shape'] and row['dtype']==aa['dtype'] and row['SHA256']==aa['SHA256'],row['name']
        self.assert_resources_unchanged()
    @staticmethod
    def packed_mapping(key,expected):
        for part,label in [('.q_proj.','q'),('.k_proj.','k'),('.v_proj.','v')]:
            if part in key:
                nq=expected[key.replace(part,'.q_proj.')]['shape'][0];nk=expected[key.replace(part,'.k_proj.')]['shape'][0]
                return key.replace(part,'.qkv_proj.'),{'q':0,'k':nq,'v':nq+nk}[label],nq+2*nk
        for part,label in [('.gate_proj.',0),('.up_proj.',1)]:
            if part in key:
                n=expected[key]['shape'][0];return key.replace(part,'.gate_up_proj.'),label*n,2*n
        return key,None,None
    def assert_resources_unchanged(self):
        for name,prior in self.original_files.items():
            st=(self.root/name).stat();assert (st.st_size,st.st_mtime_ns,st.st_ino)==prior,name
    def __getitem__(self,name):return self.values[name]

class OriginalRotary:
    def __init__(self,device):
        freq=1.0/(100000.0**(torch.arange(0,64,2,device=device,dtype=torch.float32)/64))
        positions=torch.arange(8192,device=device,dtype=torch.float32)
        angles=torch.einsum('i,j->ij',positions,freq)
        self.cache=torch.cat([angles.cos(),angles.sin()],-1).bfloat16()
        assert self.cache.shape==(8192,64)
    def apply(self,value,positions):
        c,s=self.cache[positions].chunk(2,-1);c=c[:,None].float();s=s[:,None].float();x,y=value.float().chunk(2,-1)
        return torch.cat([x*c-y*s,y*c+x*s],-1).to(value.dtype)

class GraniteDoclingNativeMath:
    def __init__(self,weights,config,observe=None):
        t,v=config['text_config'],config['vision_config']
        assert config['architectures']==['Idefics3ForConditionalGeneration'] and config['scale_factor']==4 and config['image_token_id']==100270
        assert (t['hidden_size'],t['num_hidden_layers'],t['intermediate_size'],t['num_attention_heads'],t['num_key_value_heads'],t['head_dim'],t['vocab_size'])==(576,30,1536,9,3,64,100352)
        assert (v['hidden_size'],v['num_hidden_layers'],v['intermediate_size'],v['num_attention_heads'],v['image_size'],v['patch_size'])==(768,12,3072,12,512,16)
        assert t['rms_norm_eps']==1e-5 and t['rope_parameters']=={'rope_theta':100000.0,'rope_type':'default'} and t['max_position_embeddings']==8192 and t['hidden_act']=='silu'
        assert t['tie_word_embeddings'] is False and config['tie_word_embeddings'] is False
        assert not t['attention_bias'] and not t['mlp_bias'] and t['rope_parameters']['rope_type']=='default'
        assert v['hidden_act']=='gelu_pytorch_tanh' and v['layer_norm_eps']==1e-6
        self.w,self.device,self.observe=weights,weights.device,observe
        self.rope=OriginalRotary(self.device);self.cache=[{} for _ in range(30)];self.ordinal=0
        # All original200 tiles are full512 squares with all-one masks. The
        # source bucketize(right=True) therefore gives these exact1024 IDs.
        self.vision_positions=torch.arange(1024,device=self.device,dtype=torch.int64)
    def emit(self,name,value,**meta):
        assert value.device==self.device
        if self.observe:self.observe(name,value,meta)
        return value
    def linear(self,value,name):return F.linear(value,self.w[name+'.weight'],self.w.values.get(name+'.bias'))
    def ln(self,value,name):return F.layer_norm(value,(768,),self.w[name+'.weight'],self.w[name+'.bias'],1e-6)
    @torch.inference_mode()
    def vision(self,pixels):
        assert pixels.dtype==torch.bfloat16 and pixels.device==self.device and pixels.shape[1:]==(3,512,512)
        self.emit('vision_actual_pixels',pixels)
        p='model.vision_model'
        x=F.conv2d(pixels,self.w[p+'.embeddings.patch_embedding.weight'],self.w[p+'.embeddings.patch_embedding.bias'],stride=16).flatten(2).transpose(1,2)
        x=x+F.embedding(self.vision_positions,self.w[p+'.embeddings.position_embedding.weight'])
        for i in range(12):
            n=p+'.encoder.layers.'+str(i);residual=x
            qkv=self.linear(self.ln(x,n+'.layer_norm1'),n+'.self_attn.qkv_proj')
            q,k,v=[a.reshape(a.shape[0],1024,12,64).transpose(1,2) for a in qkv.chunk(3,-1)]
            a=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0,is_causal=False,scale=.125).transpose(1,2).reshape_as(x)
            x=self.linear(a,n+'.self_attn.out_proj')+residual;residual=x
            h=self.linear(self.ln(x,n+'.layer_norm2'),n+'.mlp.fc1')
            h=self.emit('vision_activation_%02d'%i,F.gelu(h,approximate='tanh'),original_activation='gelu_pytorch_tanh')
            x=self.linear(h,n+'.mlp.fc2')+residual
        x=self.ln(x,p+'.post_layernorm');self.emit('vision_encoder_output',x)
        b=x.shape[0]
        # Exact original Idefics3Connector pixel_shuffle, preserving row/col
        # order. This is not average pooling or a replacement resampler.
        y=x.view(b,32,32,768).view(b,32,8,3072).permute(0,2,1,3).reshape(b,8,8,12288).permute(0,2,1,3).reshape(b,64,12288)
        y=self.linear(y,'model.connector.modality_projection.proj')
        assert y.shape==(b,64,576)
        return self.emit('visual_projector_output',y)
    @torch.inference_mode()
    def embeddings(self,request):
        ids=request['token_ids'];positions=request['positions'];x=F.embedding(ids,self.w['model.text_model.embed_tokens.weight'])
        # No Gemma sqrt normalization and no generated image-token suppression.
        soft_positions=request.get('image_soft_positions')
        if soft_positions is not None:
            assert len(soft_positions)==request['visuals'].shape[0]
            start,end=int(positions[0]),int(positions[-1])+1
            picks=[(i,pos-start) for i,pos in enumerate(soft_positions) if start<=pos<end]
            if picks:
                feature_ids=torch.tensor([i for i,j in picks],device=self.device,dtype=torch.int64)
                target=torch.tensor([j for i,j in picks],device=self.device,dtype=torch.int64)
                assert bool((ids[target]==100270).all())
                x[target]=request['visuals'][feature_ids]
                assert torch.equal(x[target],request['visuals'][feature_ids])
        return x
    @torch.inference_mode()
    def text_step(self,requests):
        lengths=[r['token_ids'].numel() for r in requests];assert requests and min(lengths)>0
        positions=torch.cat([r['positions'] for r in requests]);x=torch.cat([self.embeddings(r) for r in requests]);residual=None
        self.emit('text_input_embeddings',x,ordinal=self.ordinal,request_ids=[r['id'] for r in requests],lengths=lengths,positions=positions.cpu().tolist())
        for layer in range(30):
            p='model.text_model.layers.'+str(layer)
            if residual is None:residual=x;n=native_rms(x,self.w[p+'.input_layernorm.weight'])
            else:n,residual=native_rms(x,self.w[p+'.input_layernorm.weight'],residual)
            q,k,v=self.linear(n,p+'.self_attn.qkv_proj').split([576,192,192],-1)
            q=self.rope.apply(q.reshape(-1,9,64),positions);k=self.rope.apply(k.reshape(-1,3,64),positions);v=v.reshape(-1,3,64)
            chunks={};offset=0
            for r,length in zip(requests,lengths):
                rid=r['id'];ck=k[offset:offset+length];cv=v[offset:offset+length]
                chunks[rid]=(r,ck,cv)
                for tag,t in [('key',ck),('value',cv)]:self.emit('KV_delta_'+tag+'_%02d'%layer,t,request_id=rid,positions=r['positions'].cpu().tolist(),ordinal=self.ordinal,prefix_source_request_id=r.get('prefix_source_request_id'),prefix_token_count=int(r.get('prefix_token_count',0)))
                offset+=length
            pending=set(chunks)
            while pending:
                progressed=False
                for rid in list(pending):
                    r,ck,cv=chunks[rid];source=r.get('prefix_source_request_id');count=int(r.get('prefix_token_count',0))
                    if source is not None:
                        assert rid not in self.cache[layer] and count>0 and source!=rid
                        if source in pending:continue
                        sk,sv=self.cache[layer][source];assert count<=sk.shape[0];past=(sk[:count],sv[:count])
                    else:past=self.cache[layer].get(rid)
                    n=0 if past is None else past[0].shape[0]
                    assert torch.equal(r['positions'],torch.arange(n,n+ck.shape[0],device=self.device,dtype=torch.int64)),('Past/positions mismatch',rid,layer)
                    if past is not None:ck=torch.cat([past[0],ck]);cv=torch.cat([past[1],cv])
                    self.cache[layer][rid]=(ck,cv);pending.remove(rid);progressed=True
                assert progressed,'Cyclic prefix cache dependency'
            results=[];offset=0
            for r,length in zip(requests,lengths):
                ck,cv=self.cache[layer][r['id']];mask=r['positions'][:,None]>=torch.arange(ck.shape[0],device=self.device)[None]
                y=F.scaled_dot_product_attention(q[offset:offset+length].transpose(0,1)[None],ck.repeat_interleave(3,1).transpose(0,1)[None],cv.repeat_interleave(3,1).transpose(0,1)[None],attn_mask=mask[None,None],dropout_p=0.0,is_causal=False,scale=.125)
                results.append(y[0].transpose(0,1).reshape(length,576));offset+=length
                self.emit('attention_visibility_%02d'%layer,mask,request_id=r['id'],past_length=ck.shape[0]-length,key_length=ck.shape[0],layer_type='full_attention',q_heads=9,kv_heads=3)
            attention=self.linear(torch.cat(results),p+'.self_attn.o_proj')
            n,residual=native_rms(attention,self.w[p+'.post_attention_layernorm.weight'],residual)
            h=self.emit('language_activation_%02d'%layer,native_silu_mul(self.linear(n,p+'.mlp.gate_up_proj')),original_activation='silu_and_mul')
            x=self.linear(h,p+'.mlp.down_proj')
        x,_=native_rms(x,self.w['model.text_model.norm.weight'],residual)
        last=torch.tensor([sum(lengths[:i+1])-1 for i in range(len(lengths))],device=self.device,dtype=torch.int64)
        logits=F.linear(x[last],self.w['lm_head.weight'])
        assert logits.dtype==torch.bfloat16 and logits.shape==(len(requests),100352) and bool(torch.isfinite(logits).all())
        self.emit('full_vocabulary_logits',logits,ordinal=self.ordinal,request_ids=[r['id'] for r in requests],argmax=logits.argmax(-1).cpu().tolist());self.ordinal+=1
        return logits
    def release_requests(self,ids):
        for layer in self.cache:
            for rid in ids:
                assert rid in layer;del layer[rid]
