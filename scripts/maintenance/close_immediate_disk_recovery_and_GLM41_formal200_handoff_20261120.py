import sys
sys.dont_write_bytecode=True
from pathlib import Path
import hashlib,json,os,subprocess
R=Path('/root/projects/comperror');D=R/'workspace/diagnostics/batch_910b'
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def js(p):return json.loads(Path(p).read_bytes())
def put(p,x):
 t=p.with_name(p.name+'.tmp.'+str(os.getpid()));t.write_bytes((json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode());os.replace(t,p)
def main(receipt,expected,summary):
 assert sha(receipt)==expected;rec=js(receipt);pointer=D/'A100_ALIGNMENT_HANDOFF_CURRENT.json';assert sha(pointer)==rec['pointer_SHA256'];old=pointer.read_bytes();ptr=json.loads(old);H=Path(ptr['handoff_directory']);note=Path(rec['late_actual']).parent;assert summary.resolve().is_relative_to(note)
 canon={p:sha(p) for p in ptr['canonical_SHA256']};assert canon==ptr['canonical_SHA256']
 p=subprocess.run([sys.executable,'-B',str(R/'scripts/maintenance/read_immediate_disk_recovery_cleanup_and_current_GLM41_20261120.py')],capture_output=True,text=True);assert p.returncode==0,p.stderr;x=json.loads(p.stdout);assert x['watcher_files']['state.json']['value']['phase']=='formal_NN_NF_or_original_EX_active_no_cleanup' and x['watcher_files']['state.json']['value']['actual_NEW_model_forwards']==0
 assert x['current_committed_NPU'] and len(x['current_committed_NPU'])==1 and x['current_committed_NPU'][0]['completed_samples']>=105
 modelpid=int(Path(x['current_committed_NPU'][0]['home']).name.rsplit('_',1)[1]);model,=[v for v in x['formal_actual']['actual_processes'] if v['PID']==modelpid];assert any(Path(s).name=='run.py' for s in model['argv'][1:]);owner,=x['formal_actual']['original_FLOCK'];assert owner['PID']==int(model['stat'].split(') ',1)[1].split()[1])
 put(note/'latest_immediate_disk_cleanup_watcher_and_live_GLM41_formal200.json',x)
 with summary.open('a',encoding='utf-8') as f:f.write('\n最新UTC='+x['UTC']+'；即时看护PID='+str(x['watcher_actual']['PID'])+'；当前原锁PID='+str(owner['PID'])+'；正式200真实提交='+json.dumps(x['current_committed_NPU'],ensure_ascii=False)+'。\n')
 path=note/'latest_immediate_disk_cleanup_watcher_and_live_GLM41_formal200.json';close=js(H/'closing_addendum.json');close.update(UTC=x['UTC'],latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_immediate_disk_recovery_actual=str(path),latest_immediate_disk_recovery_actual_SHA256=sha(path),current_model_NN_or_NF_active=True,first_proven_current_formal_stoppage_BJT=None)
 assert pointer.read_bytes()==old and {p:sha(p) for p in canon}==canon;put(H/'closing_addendum.json',close);ptr.update(UTC=x['UTC'],closing_addendum_SHA256=sha(H/'closing_addendum.json'),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_actual_note=str(path),latest_actual_note_SHA256=sha(path),latest_actual_snapshot=str(path),latest_actual_snapshot_SHA256=sha(path),current_actual_stage='running_GLM41_same_formal200_and_noninterrupting_immediate_disk_cleanup_watch',current_actual_model_process_or_NF_active=True,current_actual_model_NN_or_NF_active=True,current_formal_model_stage='GLM41_same_approved_NEWfixed200_real_NPU4_active',first_proven_stoppage_BJT=None,severity_first_line=None)
 put(pointer,ptr);r=dict(passed=True,pointer_SHA256=sha(pointer),closing_SHA256=sha(H/'closing_addendum.json'),latest_full_actual_summary=str(summary),latest_full_actual_summary_SHA256=sha(summary),latest_actual=str(path),latest_actual_SHA256=sha(path),new_primary_pending=ptr['new_primary_pending'],new_primary_pending_SHA256=ptr['new_primary_pending_SHA256'],source_count=ptr['current_primary_source_count'],current_formal200=x['current_committed_NPU'],watcher=x['watcher_actual'],all183_canonical_unchanged=True);put(note/'immediate_cleanup_GLM41_full_handoff_receipt.json',r);print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main(Path(sys.argv[1]),sys.argv[2],Path(sys.argv[3]))
