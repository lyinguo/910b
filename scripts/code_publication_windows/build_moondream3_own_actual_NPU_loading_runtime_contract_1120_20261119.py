"""Build a no-forward NPU loading audit from the successful exact HF loader."""
from pathlib import Path
import ast,base64,hashlib,json
P=Path('D:/code-lv/.tmp/batch_910b_20261030')
s=(P/'diagnostic_moondream3_own_original_HF_actual_loading_precision_no_forward_v2_json_sets_1116_20261119.py').read_text(encoding='utf8')
s=s.replace('Moondream3_own_original_HF_actual_loading_precision_CPU_no_forward_v2_json_sets_1116_20261119','Moondream3_own_actual_NPU_loading_all662_runtime_buffers_no_forward_1120_20261119')
s=s.replace('import torch\n from tokenizers','import torch,torch_npu,numpy as np,subprocess\n device=torch.device(\'npu:4\')\n smi=subprocess.run([\'npu-smi\',\'info\'],capture_output=True,text=True,timeout=30)\n assert smi.returncode==0 and \'No running processes found in NPU 4\' in smi.stdout,smi.stdout+smi.stderr\n put(\'NPU4_admission.json\',dict(stdout=smi.stdout,stderr=smi.stderr,actual_returncode=smi.returncode))\n torch.npu.set_device(device)\n from tokenizers')
s=s.replace('state=model.state_dict();index=',"model=model.eval().to(device);torch.npu.synchronize()\n assert model.model.use_flex_decoding is True\n state=model.state_dict();index=")
s=s.replace("t.detach().contiguous().reshape(-1)","t.detach().cpu().contiguous().reshape(-1)")
s=s.replace("actual.device.type=='cpu'","actual.device==device")
s=s.replace("torch.equal(expected,actual)","torch.equal(expected,actual.detach().cpu())")
s=s.replace('actual_CPU_loaded_SHA256','actual_NPU_loaded_SHA256').replace('actual_CPU_load_casts','actual_NPU_load_casts').replace('actual_CPU_loaded_casts','actual_NPU_loaded_casts')
s=s.replace("input_and_loaded_vision_dtype_match=model.model.vision.pos_emb.dtype==torch.bfloat16,","input_and_loaded_vision_dtype_match=model.model.vision.pos_emb.dtype==torch.bfloat16,")
s=s.replace("Original own HF class actual loading/casts on CPU, no learned forward or score","Original own HF class actual loading/casts on NPU4, runtime buffers and zero KV only, no learned forward or score")
s=s.replace('no_NPU_execution=True','no_learned_NPU_execution=True')
s=s.replace('actual_NPU_loading_and_independent_full_model_NF_still_required=True','independent_full_model_NF_and_adapter_YAML_still_required=True')
needle=" dtypes=sorted({row['actual_loaded_dtype'] for row in records.values()})"
assert needle in s
block='''
 # Original construction-derived buffers: compare every value after device move.
 runtime={}
 for name,t in model.named_buffers():
  assert t.device==device,(name,t.device)
  runtime[name]=dict(shape=list(t.shape),dtype=str(t.dtype),actual_NPU_SHA256=tsha(t))
 own=importlib.import_module('universal_eval.model.moondream3_own_original_source_20261119.moondream')
 original_rope=own.precompute_freqs_cis(32,4096)
 original_mask=torch.ones((1,1,4096,4096),dtype=torch.bool).tril();original_mask[:,:,:730,:730]=True
 assert torch.equal(model.model.text.freqs_cis.detach().cpu(),original_rope)
 assert torch.equal(model.model.attn_mask.detach().cpu(),original_mask)
 assert (model.model.config.text.n_layers,model.model.config.text.dim,model.model.config.text.n_heads,model.model.config.text.n_kv_heads,model.model.config.text.max_context)==(24,2048,32,32,4096)
 model._setup_caches();cache_records={}
 assert len(model.model.text.blocks)==24
 for i,b in enumerate(model.model.text.blocks):
  for key in ['k_cache','v_cache']:
   t=getattr(b.kv_cache,key);actual=t.detach().cpu()
   assert t.device==device and t.dtype==torch.bfloat16 and tuple(t.shape)==(1,32,4096,64)
   assert torch.equal(actual,torch.zeros_like(actual))
   cache_records[f'{i}.{key}']=dict(shape=list(t.shape),dtype=str(t.dtype),actual_NPU_SHA256=tsha(t),all_original_zero_values_verified=True)
 assert len(cache_records)==48
 np.savez_compressed(H/'original_runtime_mask_RoPE_and_zero_KV.npz',original_mask=original_mask.numpy(),original_RoPE=original_rope.numpy(),zero_KV=model.model.text.blocks[0].kv_cache.k_cache.detach().cpu().view(torch.uint16).numpy())
 put('original_runtime_buffers_and_all48_physical_KV.json',dict(runtime_buffers=runtime,caches=cache_records,all_full_original_mask_values_exact=True,original_prefix_bidirectional_730=True,original_RoPE_all_values_exact=True,original_use_flex_decoding=True,all_model_forward_calls=0))
'''
s=s.replace(needle,block+needle)
s=s.replace('loading_original_own_HF_class_CPU_only_no_learned_forward','loading_original_own_HF_class_to_NPU4_no_learned_forward').replace('own_original_loaded_tensor_cast_identity_no_forward','own_original_NPU_loaded_tensor_cast_identity_no_forward')
s=s.replace('CPU placement is diagnostic only.','NPU loading is diagnostic only.')
s=s.replace('does not establish NPU dtype or a model score; it measures actual HF load casts.','establishes actual NPU loading dtype and values, without a model score or forward.')
needle=" started=time.monotonic()"
checks=""" own_files={
 'modelv2-00001-of-00004.safetensors':'79006ed488cca15b173cd5c0c7c1a467c20aaf5508e13934c36378d071d48c13',
 'modelv2-00002-of-00004.safetensors':'40202c61286ec7386d9bbce31d87af3064e42931b10323ed4b3e44158c0521e3',
 'modelv2-00003-of-00004.safetensors':'ff46835f23bac47c7409032391e02a095821e274f3faaeea3f826a960db9bf80',
 'modelv2-00004-of-00004.safetensors':'0a4d39e1bcb0ab835b9a00c7f458dedca4faf8741fc0b23fd2caf2af4547bca6',
 'tokenizer.json':'0512fdcac4a5f9e7746cbefce4a468dc93bf0f93f11e701b84f8b31bff199e9c'}
 for n,expected in own_files.items():assert sha(W/n)==expected,n
 put('fresh_own_full_weight_file_and_tokenizer_SHA256.json',own_files)
"""
assert s.count(needle)==1;s=s.replace(needle,checks+needle)
name='diagnostic_moondream3_own_actual_NPU_loading_runtime_buffers_no_forward_1120_20261119.py'
ast.parse(s);b=s.encode('utf8');(P/name).write_bytes(b)
r=dict(relative='scripts/'+name,SHA256=hashlib.sha256(b).hexdigest(),base64=base64.b64encode(b).decode())
deployment="import sys\nsys.dont_write_bytecode=True\nfrom pathlib import Path\nimport ast,base64,hashlib,runpy\nDATA="+repr(r)+"\nR=Path('/root/projects/comperror');b=base64.b64decode(DATA['base64'],validate=True);assert hashlib.sha256(b).hexdigest()==DATA['SHA256'];ast.parse(b);p=R/DATA['relative'];assert not p.exists();p.write_bytes(b);sys.path[:0]=[str(R),str(R/'scripts')];runpy.run_path(str(p),run_name='__main__')\n"
(P/'deploy_execute_Moondream3_own_actual_NPU_loading_runtime_1120_20261119_remote.py').write_text(deployment,encoding='utf8',newline='\n')
print(json.dumps({k:v for k,v in r.items() if k!='base64'}))
