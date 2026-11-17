"""Continue own completed META; fix only this own checkpoint identity binding."""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib,ast,subprocess,os,traceback
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
import scripts.prepare_continue_gemma4unified12b_QAT_own_original_BF16_sources_frontend_true3_20261117 as m
B=m.B;H=B/'Gemma4Unified12BQAT_completed_META_own_frontend_v2_native_continuation_20261117'
FRONT=B/'Gemma4Unified12BQAT_own_original_HF_false_thinking_frontend200_exact_names_v2_20261117'

def main():
    H.mkdir(exist_ok=False);m.H=H
    done=m.js(m.META/'actual_completion.json');assert done['actual_returncode']==0 and done['model_id']==m.MID
    weight_sha=next(iter(done['whole_source_SHA256'].values()))['SHA256'];assert weight_sha=='26f2cee4292298a3f9f92209643c37c80e34e011381e22434088870d9439a0a0'
    oldfront=R/'scripts/prepare_gemma4unified12b_qat_own_frontend200_exact_META_names_20261117.py';newfront=R/'scripts/prepare_gemma4unified12b_qat_own_frontend200_exact_META_names_v2_20261117.py'
    source=oldfront.read_text();assert source.count('5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d')==1
    source=source.replace('5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d',weight_sha).replace('Gemma4Unified12BQAT_own_original_HF_false_thinking_frontend200_exact_names_20261117',FRONT.name)
    ast.parse(source);assert not newfront.exists();newfront.write_text(source,encoding='utf-8');FRONT.mkdir();m.FRONT=FRONT
    oldkey=list(m.PAIRS)[1];m.PAIRS[oldkey]=str(newfront.relative_to(R));m.run(str(newfront.relative_to(R)),'own_frontend200_v2');assert m.js(FRONT/'actual_completion.json')['actual_returncode']==0
    records=[]
    for old,new in list(m.PAIRS.items())[2:]:records.append(m.derive(old,new,weight_sha))
    ready=next(v for k,v in m.PAIRS.items() if 'native_readiness' in k);path=R/ready;text=path.read_text();needle='sources=[native,verifier,';assert needle in text
    text=text.replace(needle,"sources=[native,verifier,'scripts/resume_gemma4unified12b_QAT_completed_META_own_frontend_pipeline_v2_20261117.py',",1);ast.parse(text);path.write_text(text,encoding='utf-8')
    for rec in records:rec['SHA256']=m.sha(R/rec['path'])
    m.put('derived_primary_sources_and_unweighted_origins.json',dict(records=records,own_checkpoint_SHA256=weight_sha,completed_own_META_reused_not_recomputed=True,failed_frontend0_requests_preserved=True,no_foreign_checkpoint_or_tensor_or_score_used=True));m.run(ready,'own_native_readiness')
    controller=next(v for k,v in m.PAIRS.items() if '/continue_' in k);AH=B/'Gemma4Unified12BQAT_own_original_HF_BF16_true3_capacity_fresh3_fixed200_continuation_v1_20261117';assert not AH.exists();AH.mkdir()
    with (AH/'controller_stdout.log').open('xb') as out,(AH/'controller_stderr.log').open('xb') as err:child=subprocess.Popen([sys.executable,'-B',str(R/controller)],cwd=R,env=m.ENV,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
    m.put('actual_completion.json',dict(UTC=m.now(),actual_returncode=0,own_frontend200_v2_and_source_config_passed=True,launched_controller_PID=child.pid,actual_controller=str(R/controller),actual_controller_SHA256=m.sha(R/controller),subsequent_NPU_and_numeric_and_capacity_must_be_checked=True,no_formal200_or_G_or_qualification_claimed=True));m.put('state.json',dict(UTC=m.now(),PID=os.getpid(),phase='handed_to_original_private_true3_pipeline',controller_PID=child.pid,controller_home=str(AH)))

if __name__=='__main__':
    try:main()
    except BaseException:
        if H.exists() and not (H/'actual_completion.json').exists():m.put('actual_completion.json',dict(UTC=m.now(),actual_returncode=1,error=traceback.format_exc(),no_new_G_or_qualification=True))
        raise
