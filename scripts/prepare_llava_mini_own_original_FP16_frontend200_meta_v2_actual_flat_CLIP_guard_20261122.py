"""Own fixed200 tokens/pixels and full checkpoint/meta contract; no learned forward."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
from decimal import Decimal
import ast,contextlib,hashlib,inspect,json,os,struct,traceback,zipfile
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
sys.path[:0]=[str(R),str(R/'scripts')]
MID='ICTNLP/llava-mini-llama-3.1-8b'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/LlavaMiniLlamaForCausalLM')/MID
H=B/'LlavaMini_own_original_FP16_frontend200_meta_v2_actual_flat_CLIP_guard_no_learned_NN_20261122'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def now():return datetime.now(timezone.utc).isoformat()
def put(n,v):
    p=H/n;t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8');os.replace(t,p)
def digest(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def main():
    import numpy as np
    import torch
    from PIL import Image
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer,PreTrainedTokenizerFast,CLIPImageProcessor
    from torchvision.transforms.v2 import functional as tv
    from torchvision.transforms import InterpolationMode
    from transformers.cache_utils import DynamicLayer
    from universal_eval.model.adapters import high_nine_llava_reference as own
    from universal_eval.model.llava_mini_own_saved_graph_FP16_20261122 import build_own_saved_graph,original_text_kwargs
    from universal_eval.orchestrator.pipeline import DatasetLoader,TaskDataRegistry,DataPreprocessor
    from universal_eval.algorithm.inference import GreedyInference
    import xml.etree.ElementTree as ET
    from batch_910b_support import workbook_rows
    torch.set_num_threads(4);H.mkdir(exist_ok=False)
    original=R/'universal_eval/model/adapters/high_nine_llava_reference.py'
    assert sha(original)=='309134ea42aeecbc317f59ec4b81ad068dc21b2856237c49e80dcb23f8682226'
    saved=R/'configs/LlavaMiniLlamaForCausalLM/ICTNLP--llava-mini-llama-3.1-8b.yaml'
    assert sha(saved)=='f133a3fd9341c12ba7c1e6867711682b04685e5184108ed2148da407cee67321'
    before_cache=inspect.getsource(DynamicLayer.lazy_initialization)
    raw=js(W/'config.json');index=js(W/'model.safetensors.index.json')['weight_map'];headers={};files={}
    for name in sorted(set(index.values())):
        p=W/name
        with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(n))
        keys={k for k in header if k!='__metadata__'};assert keys=={k for k,v in index.items() if v==name}
        files[name]=dict(bytes=p.stat().st_size,SHA256=sha(p))
        for k in keys:
            v=header[k];assert k not in headers;headers[k]=dict(source_file=name,shape=v['shape'],source_dtype=v['dtype'],data_offsets=v['data_offsets'])
    assert len(headers)==732 and set(headers)==set(index)
    put('own_weight_file_SHA_header_checkpoint.json',dict(UTC=now(),own_weight_files=files,headers=headers,source_tensor_count=732,only_source_and_header_checked_not_NN_or_full_meta_acceptance=True))
    # Compare the copied configuration expression to the actual saved adapter AST.
    oldtree=ast.parse(original.read_text());load=next(n for n in ast.walk(oldtree) if isinstance(n,ast.FunctionDef) and n.name=='load')
    oldkw=next(n.value for n in ast.walk(load) if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name) and n.target.id=='cfg_kwargs')
    newkw=next(n.value for n in ast.walk(ast.parse(inspect.getsource(original_text_kwargs))) if isinstance(n,ast.Return))
    assert ast.dump(oldkw,include_attributes=False)==ast.dump(newkw,include_attributes=False)
    model=build_own_saved_graph(raw,meta=True);meta=model.state_dict();assert set(meta)==set(headers),(sorted(set(meta)-set(headers)),sorted(set(headers)-set(meta)))
    assert all(list(meta[k].shape)==v['shape'] for k,v in headers.items())
    assert model.lm_head.weight.shape==(128256,4096) and len(model.model.layers)==32 and len(model.model.prefusion_layers)==4
    assert len(model.model.vision_tower.vision_tower.vision_model.encoder.layers)==24
    assert model.config.rope_parameters['rope_theta']==500000.0 and model.config.rope_parameters['rope_type']=='default'
    source_dtype_counts={t:sum(v['source_dtype']==t for v in headers.values()) for t in {v['source_dtype'] for v in headers.values()}}
    put('own_source_header_manifest.json',headers)
    put('own_source_meta_review.json',dict(passed=True,UTC=now(),model_id=MID,own_weight_files=files,source_tensor_count=732,source_dtype_counts=source_dtype_counts,all732_meta_names_shapes_exact=True,original_explicit_runtime_load_cast_is_FP16_but_actual_NPU_unproved=True,original_saved_plain_RoPE_branch_preserved=True,raw_checkpoint_rope_scaling_not_silently_restored=raw['rope_scaling'],original_max_new_tokens=2048,original_batch_size=1,original_vision24_prefusion4_text32_full_head128256=True,no_learned_CPU_or_NPU_forward=True))
    del model,meta
    try:tk=AutoTokenizer.from_pretrained(W,local_files_only=True);first_error=None
    except Exception as err:first_error=type(err).__name__+': '+str(err);tk=PreTrainedTokenizerFast.from_pretrained(W,local_files_only=True)
    if tk.pad_token is None:tk.pad_token=tk.eos_token
    assert len(tk)==128256
    actor=own.LlavaMiniCUDAAdapter(device='cpu',max_new_tokens=2048,torch_dtype='float16');actor._tokenizer=tk;actor._mm_use_im_start_end=bool(raw.get('mm_use_im_start_end',False))
    processor=CLIPImageProcessor(size={'shortest_edge':336},crop_size={'height':336,'width':336},do_resize=True,do_center_crop=True,do_rescale=True,do_normalize=True,image_mean=own.CLIP_MEAN,image_std=own.CLIP_STD)
    rust=Tokenizer.from_file(str(W/'tokenizer.json'))
    dataset=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl';afs=Path('/mnt/afs/comp_error/workspace/datasets/image-text-to-text-mmbench/eval.jsonl')
    assert sha(dataset)==sha(afs)
    def rows(p,flt):
        x=DatasetLoader().load(str(p));x=TaskDataRegistry._resolve_media_paths(x,str(p.parent));return DataPreprocessor().preprocess(x,task_type='image-text-to-text',apply_template=False,filter_empty=flt)
    records=rows(dataset,False);filtered=rows(dataset,True);original_rows=rows(afs,True)
    assert len(records)==len(filtered)==len(original_rows)==4329
    assert [(x['prompt'],x['reference']) for x in records]==[(x['prompt'],x['reference']) for x in filtered]==[(x['prompt'],x['reference']) for x in original_rows]
    mean=np.asarray(own.CLIP_MEAN,dtype=np.float32)[:,None,None]*np.float32(255)
    std=np.asarray(own.CLIP_STD,dtype=np.float32)[:,None,None]*np.float32(255)
    front=[]
    for i,row in enumerate(records[:200]):
        assert sha(row['media_path'])==sha(original_rows[i]['media_path'])
        prompt=GreedyInference._add_answer_instruction(row['prompt']);text=actor._build_prompt_text(prompt);ids=actor._tokenizer_image_token(text).tolist()
        chunks=[rust.encode(c,add_special_tokens=True).ids for c in text.split('<image>')]
        offset=int(bool(chunks and chunks[0] and chunks[0][0]==tk.bos_token_id));ind=[chunks[0][0]] if offset else []
        for j,c in enumerate(chunks):
            if j:ind.extend(([-200]*(offset+1))[offset:])
            ind.extend(c[offset:])
        assert ind==ids and ids.count(-200)==1
        with Image.open(row['media_path']) as im:image=im.convert('RGB')
        native=processor(images=image,return_tensors='pt')['pixel_values'];assert native.dtype==torch.float32 and list(native.shape)==[1,3,336,336]
        rgb=np.asarray(image,dtype=np.uint8).copy();t=torch.from_numpy(rgb.transpose(2,0,1).copy())
        h,w=t.shape[-2:];short=min(h,w);nh,nw=(336,int(336*w/h)) if h<=w else (int(336*h/w),336)
        resized=tv.resize(t,[nh,nw],interpolation=InterpolationMode.BICUBIC,antialias=True)
        top=int((nh-336)/2.0);left=int((nw-336)/2.0);u=resized[:,top:top+336,left:left+336].numpy().copy()
        independent=((u.astype(np.float32)-mean)/std)[None]
        assert np.array_equal(independent,native.numpy()) and digest(independent)==digest(native.numpy()),('Full independent own F32 pixel mismatch',i,float(np.max(np.abs(independent-native.numpy()))))
        half=independent.astype(np.float16);actual_half=native.to(torch.float16).numpy();assert np.array_equal(half.view(np.uint16),actual_half.view(np.uint16))
        front.append(dict(index=i,prompt=row['prompt'],reference=row['reference'],media_path=row['media_path'],media_file_SHA256=sha(row['media_path']),actual_model_prompt=prompt,text=text,input_ids=ids,original_image_placeholder=-200,original_add_generation_prompt_false_preserved=True,independent_Rust_all_token_IDs_exact=True,pixels_F32_shape=list(native.shape),pixels_F32_SHA256=digest(independent),pixels_FP16_SHA256=digest(half),independent_TV2_uint8_bicubic_and_numpy_F32_fused_normalization_full_values_exact=True))
        if i in [0,99,199]:
            from universal_eval.utils.complete_standard_LZMA_NPZ_values_20261118 import write_complete_npz
            write_complete_npz(H/('input_example_%05d.npz'%i),dict(input_ids=np.asarray(ids,dtype=np.int64),pixels_F32=independent,pixels_FP16=half,resized_uint8=u))
        put('progress.json',dict(UTC=now(),phase='own_input_only_frontend_no_learned_CPU_or_NPU',completed=i+1,total=200))
    assert inspect.getsource(DynamicLayer.lazy_initialization)==before_cache
    put('own_original_frontend200.json',front)
    book=Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx');binding=js(D/'A100_ALIGNMENT_LOCAL_WORKBOOK.json');assert sha(book)==binding['local_workbook_sha256']
    allrows,_,_=workbook_rows(book);row,=[x for x in allrows if x['model_id']==MID];assert row['task']=='image-text-to-text' and row['score'] is None
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(book) as z:cell=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).find('.//s:c[@r="D'+str(row['row'])+'"]',ns);server_D=cell.findtext('s:v',namespaces=ns)
    old=Path(binding['local_A100_reference_snapshot']);current=Path(binding['current_user_workbook_D_reference_snapshot']);assert sha(old)==binding['local_A100_reference_snapshot_sha256'] and sha(current)==binding['current_user_workbook_D_reference_snapshot_sha256']
    three=dict(server=server_D,local=js(current)['models'][MID]['A100_D'],preserved_original=js(old)['models'][MID]['A100_D']);assert all(Decimal(x)==Decimal('.745') for x in three.values())
    assets={p.name:sha(p) for p in W.iterdir() if p.is_file() and p.name in ['config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json']}
    source=['scripts/prepare_llava_mini_own_original_FP16_frontend200_meta_v2_actual_flat_CLIP_guard_20261122.py','universal_eval/model/llava_mini_own_saved_graph_FP16_20261122.py','universal_eval/model/adapters/high_nine_llava_reference.py','universal_eval/algorithm/inference.py','run.py','universal_eval/orchestrator/pipeline.py']
    value=dict(passed=True,UTC=now(),model_id=MID,own_weight_path=str(W),own_weight_files=files,own_assets_SHA256=assets,source_tensor_count=732,source_dtype_counts=source_dtype_counts,own_source_meta_review=str(H/'own_source_meta_review.json'),own_source_meta_review_SHA256=sha(H/'own_source_meta_review.json'),own_frontend200=str(H/'own_original_frontend200.json'),own_frontend200_SHA256=sha(H/'own_original_frontend200.json'),own_source_header_manifest=str(H/'own_source_header_manifest.json'),own_source_header_manifest_SHA256=sha(H/'own_source_header_manifest.json'),dataset_path=str(dataset),dataset_SHA256=sha(dataset),original_AFS_dataset_same_bytes_and_all4329_prompt_reference_order_and_200_images_verified=True,three_D_values=three,current_row=row['row'],original_saved_config=str(saved),original_saved_config_SHA256=sha(saved),source_sha256={n:sha(R/n) for n in source},original_explicit_load_FP16_budget2048_outer1_full_image_compressor1_prefusion4_text32_and_plain_RoPE_preserved=True,actual_NPU_model_dtype_not_yet_proved=True,all200_own_full_F32_pixels_FP16_cast_tokens_exact=True,original_tokenizer_fallback_error=first_error,shared_cache_library_unmodified=True,ready_for_NPU=False,missing_next_stage='Complete own NPU model observations/full independent verifier/standalone execution binding before genuine NEW3; no learned CPU or NPU forward yet',no_A100_historical_runtime_or_tensor_capture_claim=True)
    import PIL.Image as pil_image,tokenizers.tokenizers as rust_binary,transformers.image_processing_backends as image_backend,torchvision.transforms.v2.functional as tv_backend
    import transformers.models.llama.modeling_llama as llama_backend,transformers.models.clip.modeling_clip as clip_backend
    direct_external=[Path(m.__file__) for m in [pil_image,rust_binary,image_backend,tv_backend,llama_backend,clip_backend,np,torch]]
    value['input_only_external_runtime_sources_sha256']={str(p):sha(p) for p in direct_external}
    put('readiness.json',value);put('actual_completion.json',dict(UTC=value['UTC'],actual_returncode=0,no_learned_CPU_or_NPU_forward=True));return value

if __name__=='__main__':
    try:
        with contextlib.redirect_stdout(sys.stderr):value=main()
        print(json.dumps(value,ensure_ascii=False))
    except BaseException:
        if H.exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),all_original_source_assets_reports_logs_unchanged=True,no_learned_CPU_or_NPU_forward=True))
        raise
