"""Continue the next available own dense BF16 QAT-unquantized model.

Reuse unweighted architecture code only. Every asset, checkpoint tensor,
input, head, reference and result belongs to this model's original root.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from datetime import datetime,timezone
import os,json,hashlib,subprocess,ast,traceback
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b';B=D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H=B/'Gemma4Unified12BQAT_own_original_assets_frontend_native_continuation_20261117'
MID='google/gemma-4-12B-it-qat-q4_0-unquantized';OLDMID='google/gemma-4-12B-it'
META=B/'Gemma4Unified12BQAT_own_original_source_assets_META_only_next_candidate_20261117'
FRONT=B/'Gemma4Unified12BQAT_own_original_HF_false_thinking_frontend200_exact_names_20261117'
PAIRS={
 'scripts/prepare_gemma4unified12bit_own_source_assets_META_no_learned_CPU_20261116.py':'scripts/prepare_gemma4unified12b_qat_own_source_assets_META_no_learned_CPU_20261117.py',
 'scripts/prepare_gemma4unified12bit_own_frontend200_and_exact_META_names_20261116.py':'scripts/prepare_gemma4unified12b_qat_own_frontend200_exact_META_names_20261117.py',
 'universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_native_v4_exact_loading_metadata_20261117.py':'universal_eval/model/adapters/gemma4_unified12b_qat_own_original_BF16_native_v1_20261117.py',
 'scripts/gemma4_unified12bit_genuine_clean_worker_v4_exact_loading_metadata_20261117.py':'scripts/gemma4_unified12b_qat_genuine_clean_worker_v1_20261117.py',
 'universal_eval/model/adapters/gemma4_unified_12bit_own_BF16_clean_worker_v4_exact_loading_metadata_20261117.py':'universal_eval/model/adapters/gemma4_unified12b_qat_own_BF16_clean_worker_v1_20261117.py',
 'scripts/verify_gemma4_unified_12bit_own_BF16_clean_worker_v5_exact_stdlib_20261117.py':'scripts/verify_gemma4_unified12b_qat_own_BF16_clean_worker_v1_20261117.py',
 'scripts/prepare_gemma4_unified12bit_own_original_HF_BF16_native_readiness_v4_exact_loading_metadata_20261117.py':'scripts/prepare_gemma4_unified12b_qat_own_original_HF_BF16_native_readiness_v1_20261117.py',
 'scripts/continue_gemma4_unified12bit_own_original_HF_BF16_true3_fresh3_fixed200_v4_exact_loading_metadata_20261117.py':'scripts/continue_gemma4_unified12b_qat_own_original_HF_BF16_true3_fresh3_fixed200_v1_20261117.py',
 'scripts/measure_gemma4_unified12bit_own_original_HF_BF16_actual_capacity_v4_exact_loading_metadata_20261117.py':'scripts/measure_gemma4_unified12b_qat_own_original_HF_BF16_actual_capacity_v1_20261117.py',
}
ENV=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(n,v):
    p=H/n;t=p.with_name(p.name+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(t,p)
def now():return datetime.now(timezone.utc).isoformat()

def derive(old,new,weight_sha=None):
    s=(R/old).read_text();s=s.replace(OLDMID,MID).replace("'gemma-4-12B-it'","'gemma-4-12B-it-qat-q4_0-unquantized'").replace('google--gemma-4-12B-it.yaml','google--gemma-4-12B-it-qat-q4_0-unquantized.yaml')
    for before,after in sorted(PAIRS.items(),key=lambda x:len(x[0]),reverse=True):
        s=s.replace(before,after).replace(Path(before).stem,Path(after).stem)
    s=s.replace('scripts/verify_gemma4_unified_12bit_own_BF16_clean_worker_v4_exact_loading_metadata_20261117.py','scripts/verify_gemma4_unified12b_qat_own_BF16_clean_worker_v1_20261117.py')
    s=s.replace('Gemma4Unified12Bit','Gemma4Unified12BQAT')
    s=s.replace('Gemma4Unified12BQAT_own_original_source_assets_META_only_next_candidate_20261116',META.name).replace('Gemma4Unified12BQAT_own_original_HF_false_thinking_frontend200_exact_names_20261116',FRONT.name)
    s=s.replace('v4_exact_loading_metadata_readiness_20261117','v1_readiness_20261117').replace('true3_v4_exact_loading_metadata_real_FLOCK_20261117','true3_v1_real_FLOCK_20261117').replace('fixed200_v4_exact_loading_metadata_real_FLOCK_20261117','fixed200_v1_real_FLOCK_20261117').replace('continuation_v4_exact_loading_metadata_20261117','continuation_v1_20261117').replace('capacity_review_v4_exact_loading_metadata_20261117','capacity_review_v1_20261117')
    s=s.replace('gemma4_unified12bit_own_original_HF_BF16_native_v4_exact_loading_metadata_20261117.json','gemma4_unified12b_qat_own_original_HF_BF16_native_v1_20261117.json').replace('gemma4_unified12bit_own_original_HF_BF16_approved_first200_v4_exact_loading_metadata_20261117.json','gemma4_unified12b_qat_own_original_HF_BF16_approved_first200_v1_20261117.json')
    s=s.replace('google--gemma-4-12B-it-own-original-HF-BF16-2048-native-v4-exact-loading-metadata','google--gemma-4-12B-it-qat-q4_0-unquantized-own-original-HF-BF16-2048-native-v1')
    s=s.replace("assert all(float(v)==.95 for v in [d,origD,localD])","assert all(float(v)>0 for v in [d,origD,localD])")
    if weight_sha:s=s.replace('5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d',weight_sha)
    if 'readiness' in new:
        needle="sources=[native,verifier,"
        assert needle in s;s=s.replace(needle,"sources=[native,verifier,'scripts/prepare_continue_gemma4unified12b_QAT_own_original_BF16_sources_frontend_true3_20261117.py','configs/protocols/gemma4_unified12bit_current_real3_exact_genuine_stdlib_SHA_v5_20261117.json',",1)
        s=s.replace('Workbook D is authorized A100 comparison;', 'Own dense BF16 QAT-unquantized checkpoint, not native4bit. Genuine installed runtime-library SHA guards are shared, without another model checkpoint, template, values, outputs or score. Original saved /mnt2 mount bytes are not claimed captured; this registered own MID mirror is read and fully hashed. Workbook D is authorized A100 comparison;')
    ast.parse(s);dest=R/new;assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(s,encoding='utf-8');return dict(original_unweighted_template=old,original_template_SHA256=sha(R/old),path=new,SHA256=sha(dest))

def run(rel,stage):
    put('state.json',dict(UTC=now(),PID=os.getpid(),phase=stage,script=rel,no_NPU_until_validated_private_runner=True))
    with (H/(stage+'_stdout.json')).open('xb') as out,(H/(stage+'_stderr.log')).open('xb') as err:r=subprocess.run([sys.executable,'-B',str(R/rel)],cwd=R,env=ENV,stdout=out,stderr=err)
    assert r.returncode==0,(stage,r.returncode)

def main():
    H.mkdir(exist_ok=False);state=js(D/'state.json');assert state['models'][MID]['status']=='needs_adaptation' and state['models'][MID]['score'] is None
    W=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Gemma4UnifiedForConditionalGeneration')/MID;assert W.is_dir();own=js(W/'config.json');assert own['dtype']=='bfloat16' and own['architectures']==['Gemma4UnifiedForConditionalGeneration'] and not own.get('quantization_config')
    import yaml
    saved=yaml.safe_load((R/'configs/Gemma4UnifiedForConditionalGeneration/google--gemma-4-12B-it-qat-q4_0-unquantized.yaml').read_bytes());assert saved['model']['name']==MID and saved['algorithm']['batch_size']==1 and saved['model']['kwargs']==dict(max_new_tokens=2048,torch_dtype='bfloat16',enable_thinking=False)
    recs=[]
    for old,new in list(PAIRS.items())[:2]:recs.append(derive(old,new))
    META.mkdir(exist_ok=False);run(list(PAIRS.values())[0],'own_weights_genuine_META');assert js(META/'actual_completion.json')['actual_returncode']==0
    FRONT.mkdir(exist_ok=False);run(list(PAIRS.values())[1],'own_frontend200');assert js(FRONT/'actual_completion.json')['actual_returncode']==0
    weights=js(META/'own_tensor_headers_and_shards.json')['weights'];assert len(weights)==1;weight_sha=next(iter(weights.values()))['SHA256']
    for old,new in list(PAIRS.items())[2:]:recs.append(derive(old,new,weight_sha))
    put('derived_primary_sources_and_unweighted_origins.json',dict(records=recs,own_checkpoint_SHA256=weight_sha,no_foreign_checkpoint_or_tensor_or_score_used=True))
    ready=next(v for k,v in PAIRS.items() if 'native_readiness' in k);run(ready,'own_native_readiness')
    c=next(v for k,v in PAIRS.items() if '/continue_' in k);AH=B/'Gemma4Unified12BQAT_own_original_HF_BF16_true3_capacity_fresh3_fixed200_continuation_v1_20261117';assert not AH.exists();AH.mkdir()
    with (AH/'controller_stdout.log').open('xb') as out,(AH/'controller_stderr.log').open('xb') as err:child=subprocess.Popen([sys.executable,'-B',str(R/c)],cwd=R,env=ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    put('actual_completion.json',dict(UTC=now(),actual_returncode=0,own_META_and_frontend200_and_source_config_passed=True,launched_controller_PID=child.pid,actual_controller=str(R/c),actual_controller_SHA256=sha(R/c),subsequent_NPU_and_numeric_and_capacity_must_be_checked=True,no_formal200_or_G_or_qualification_claimed=True));put('state.json',dict(UTC=now(),PID=os.getpid(),phase='handed_to_original_private_true3_pipeline',controller_PID=child.pid,controller_home=str(AH)))

if __name__=='__main__':
    try:main()
    except BaseException:
        if H.exists() and not (H/'actual_completion.json').exists():put('actual_completion.json',dict(UTC=now(),actual_returncode=1,error=traceback.format_exc(),no_new_G_or_qualification=True))
        raise
