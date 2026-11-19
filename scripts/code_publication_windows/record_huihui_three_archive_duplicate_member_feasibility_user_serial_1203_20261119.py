"""Read accepted ZIP-member metadata only; no NN, array rewrite or deletion."""
from pathlib import Path
import collections,hashlib,json
P=Path(__file__).resolve().parent
source=P/'R_HuihuiGLM46_own2048_full5_fullNF_completed_capacity_actual_2048_20261118.json'
j=json.loads(source.read_bytes());cap=j['actual_states']['H/actual_completion.json']['value']['capacity']
archives=collections.defaultdict(list)
for row in cap['all_complete_ZIP_members']:archives[row['archive']].append(row)
records=[];old=0;distinct=0
for path,rows in archives.items():
 groups=collections.defaultdict(list)
 for row in rows:groups[(row['CRC'],row['bytes'])].append(row)
 before=sum(r['compressed_bytes'] for r in rows)
 after=sum(min(r['compressed_bytes'] for r in values) for values in groups.values())
 old+=before;distinct+=after
 records.append(dict(archive=path,members=len(rows),distinct_CRC_size_classes=len(groups),compressed_member_bytes=before,optimistic_distinct_compressed_member_bytes=after))
assert len(records)==3 and sum(x['members'] for x in records)==333
result=dict(scope='Read accepted complete ZIP member metadata, not new inference, fullNF, or adopted storage',source_metadata_SHA256=hashlib.sha256(source.read_bytes()).hexdigest(),archives=records,original_compressed_member_bytes=old,optimistic_after_direct_complete_NPY_sharing_bytes=distinct,optimistic_ratio=distinct/old,all333_complete_NPY_members_distinct_within_own_archive=old==distinct,no_capacity_saving_demonstrated=True,no_array_read_or_deletion=True,actual_model_forwards=0)
(P/'Huihui_completed3_ZIP_member_duplicate_feasibility_user_serial_1203_20261119.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
