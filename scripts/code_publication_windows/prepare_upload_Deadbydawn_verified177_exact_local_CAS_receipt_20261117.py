import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast
P=Path(__file__).parent
s=(P/'prepare_upload_Coder176_exact_local_receipt_20261117.py').read_text(encoding='utf8').replace('CoderGemma4E4B','DeadbydawnGemma4E2B').replace('verified176','verified177').replace('G212','G239').replace('upload_Coder176','upload_Deadbydawn177')
ast.parse(s)
exec(compile(s,str(P/'prepare_upload_Deadbydawn_verified177_exact_local_CAS_receipt_20261117.py'),'exec'),{'__file__':__file__,'__name__':'__main__'})
