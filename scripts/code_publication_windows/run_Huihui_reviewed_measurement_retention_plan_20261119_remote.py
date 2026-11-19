import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
R=Path('/root/projects/comperror');sys.path[:0]=[str(R),str(R/'scripts')]
from maintenance import retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119 as m
assert m.sha(R/'scripts/maintenance/retain_Huihui_numerically_verified200_correct_nonexamples_all_wrong_controls_preserved_20261119.py')=='5120db9f35af929f22b01610b7d27353322cdac901e9acc479292c006af9387b'
cfg=m.js(R/'configs/retention/verified180_accepted_nonexample_user_serial_continue_owner_reference_v2_20261119.json')
cfg.update(measurement_target_audit_SHA256='b831aacc983f0a03c5aaa2cafa20e5853b7ff8ba0b51a6538027177054a85d36',max_remove_bytes=10617515541,max_remove_files=105,all61_incorrect_cases_and_original_full_outer_examples_protected=True,latest_actual_user_authorization={'source':'Direct latest human user in original910b chat','latest_user_message_verbatim':'继续串行跑，Huihui完整验收通过后同步两份表，马上接下一款有权重、未适配或误差超过5%的模型，不等定期检查。完整报告、源码、配置、日志和必要数值例证保留，停摆或报错及时告诉我。','prior_cleanup_message_verbatim':'那就再检查服务器上已验收的中间矩阵和临时文件，完整报告、源码、配置、日志和必要数值例证保留，不再需要的清理掉，腾够空间马上串行继续跑，别一直等着。','scope':'Only full-NF0 Huihui200 correct nonexample NPZ; baselinegap qualification stays false, all61 wrong cases, all small/fresh3/failures and original full first/mid/last groups retained'})
assert not m.CFG.exists();m.put(m.CFG,cfg)
print(json.dumps(m.plan('b831aacc983f0a03c5aaa2cafa20e5853b7ff8ba0b51a6538027177054a85d36'),ensure_ascii=False))
