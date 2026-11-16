"""Own MedraN E4B original FP16 checkpoint and chat frontend200, with independent pixel replay.

No learned CPU forward, online A100 prerequisite or canonical score write.
"""
import sys
sys.dont_write_bytecode=True
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,struct,time,traceback

R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
MID='nicoboss/MedraN-E4B-Uncensored-EP7'
W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma3nForConditionalGeneration/nicoboss/MedraN-E4B-Uncensored-EP7')
DATA=R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'
H=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct/MedraNE4B_own_original_FP16_chat_frontend200_assets_20261116'

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(name,value):
    path=H/name;temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(temp,path)

def clean(s):
    s=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',s)
    s=re.sub(r'[ \t]+',' ',s);s=re.sub(r'\n{3,}','\n\n',s).strip()
    return re.sub(r'<image\d+>\s*','',s)

def main():
    import torch,numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from torchvision.transforms.v2 import functional as VF
    from torchvision.transforms import InterpolationMode
    from universal_eval.algorithm.inference import GreedyInference
    torch.set_num_threads(4)
    protected=[D/'state.json',D/'registry.json',Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')]
    before={str(p):sha(p) for p in protected};start=time.monotonic()
    cfg=json.loads((W/'config.json').read_bytes());gen=json.loads((W/'generation_config.json').read_bytes());pre=json.loads((W/'preprocessor_config.json').read_bytes())
    assert cfg['torch_dtype']=='float16' and cfg['text_config']['torch_dtype']=='float16' and cfg['architectures']==['Gemma3nForConditionalGeneration'] and cfg['text_config']['num_hidden_layers']==35 and cfg['vision_config']['hidden_size']==2048
    assert gen['eos_token_id']==[1,106] and gen['cache_implementation']=='hybrid'
    assert pre['size']=={'height':768,'width':768} and not pre['do_normalize'] and pre['do_rescale'] and pre['resample']==2
    index=json.loads((W/'model.safetensors.index.json').read_bytes())['weight_map'];shards=sorted(set(index.values()));assert len(shards)==4
    headers={};source={}
    for shard in shards:
        f=W/shard
        with f.open('rb') as h:
            prefix=h.read(8);length=struct.unpack('<Q',prefix)[0];assert 0<length<16*1024**2;raw=h.read(length)
        header=json.loads(raw);keys=set(header)-{'__metadata__'};assert keys=={k for k,v in index.items() if v==shard}
        data_size=f.stat().st_size-8-length
        for k in keys:
            rec=header[k];assert rec['dtype']=='F16' and 0<=rec['data_offsets'][0]<rec['data_offsets'][1]<=data_size
            assert k not in source;source[k]=dict(rec,shard=shard)
        headers[shard]=dict(bytes=f.stat().st_size,SHA256=sha(f),header_SHA256=hashlib.sha256(prefix+raw).hexdigest(),tensor_count=len(keys))
        write('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_checkpoint_full_file_SHA',checked_shards=len(headers),total_shards=4,no_NN=True))
    assets={f.name:sha(f) for f in W.iterdir() if f.is_file() and f.suffix in ['.json','.model','.jinja']}
    processor=AutoProcessor.from_pretrained(str(W),local_files_only=True,trust_remote_code=False)
    assert processor.image_processor.size=={'height':768,'width':768}
    rows=[json.loads(line) for line in DATA.read_text().splitlines()];assert len(rows)==4329
    records=[];output=H/'frontend200.jsonl';assert not output.exists()
    with output.open('x',encoding='utf-8') as dst:
        for i,row in enumerate(rows[:200]):
            prompt=clean(row['prompt']);reference=row['reference'];media=Path(row.get('media_path') or row.get('image_path') or row.get('image'))
            if not media.is_absolute():media=DATA.parent/media
            media=media.resolve();assert media.is_relative_to(DATA.parent) and media.is_file()
            image=Image.open(media).convert('RGB');model_prompt=GreedyInference._add_answer_instruction(prompt);messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':model_prompt}]}];text=processor.apply_chat_template(messages,add_generation_prompt=True,tokenize=False)
            assert text.endswith('<start_of_turn>model\n')
            inputs=processor(images=[image],text=text,return_tensors='pt',padding=True)
            # Independent original-type uint8 resize then FP32 rescale; never a model forward.
            pix=VF.pil_to_tensor(image);pix=VF.resize(pix,[768,768],interpolation=InterpolationMode.BILINEAR,antialias=True);pix=pix.to(torch.float32).mul(pre['rescale_factor'])[None]
            assert torch.equal(inputs['pixel_values'],pix),(i,'own processor/resizing operation sequence differs')
            ids=inputs['input_ids'];assert (ids==cfg['image_token_id']).sum().item()==256 and ids[0,0].item()==2
            decoded=processor.tokenizer.decode(ids[0].tolist(),skip_special_tokens=False)
            actual={}
            for name,value in inputs.items():
                value=value.detach().to(torch.float16).contiguous() if value.is_floating_point() else value.detach().contiguous();actual[name]=dict(shape=list(value.shape),dtype=str(value.dtype),SHA256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())
            rec=dict(index=i,prompt=prompt,reference=reference,media_path=str(media),image_SHA256=sha(media),original_model_prompt=model_prompt,formatted_text=text,inputs=actual,generation=dict(max_new_tokens=2048,do_sample=False),own_template_used=True,independent_pixel_replay_equal=True,historical_A100_tensor_capture_claimed=False)
            dst.write(json.dumps(rec,ensure_ascii=False)+'\n');dst.flush();records.append(rec)
            write('progress.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_CPU_chat_frontend_and_independent_pixels',completed=len(records),target=200,no_NN=True,no_A100_qualification=True))
    assert before=={str(p):sha(p) for p in protected}
    result=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,phase='own_chat_assets_and_independent_frontend_ready_no_NPU_claim',samples=200,own_checkpoint_tensor_count=len(source),own_original_storage_dtype='F16',own_original_execution_dtype='FP16',BF16_from_another_model_not_substituted=True,own_weight_shards=headers,own_runtime_assets_SHA256=assets,dataset_SHA256=sha(DATA),frontend_jsonl_SHA256=sha(output),all200_independent_F32_pixel_replays_equal=True,own_EOS_ids=gen['eos_token_id'],own_language_layers=35,own_vision_image_size=768,own_vision_soft_tokens=256,historical_A100_deployment_qualified=False,new_NPU_inference=False,new_NPU_score=None,no_G_write=True,original_canonical_SHA256=before,elapsed_seconds=time.monotonic()-start,next='Bind saved own configuration and workbook D with independent NPU module/YAML; use own checkpoint; CPU preparation is not an NPU score')
    write('source_tensor_headers.json',source);write('review.json',result);write('actual_completion.json',dict(passed=True,actual_returncode=0,UTC=result['UTC'],no_NN=True));print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    H.mkdir(parents=True,exist_ok=True)
    try:main()
    except BaseException:
        write('actual_completion.json',dict(passed=False,actual_returncode=1,UTC=datetime.now(timezone.utc).isoformat(),error=traceback.format_exc(),no_NN=True));raise
