import sys
sys.dont_write_bytecode = True
from pathlib import Path
import ast
P = Path(__file__).parent
s = (P / 'prepare_upload_Coder176_exact_local_receipt_20261117.py').read_text(encoding='utf8')
s = s.replace('CoderGemma4E4B', 'Granite41').replace('verified176', 'verified178').replace('G212', 'G364').replace('upload_Coder176', 'upload_Granite41_178')
ast.parse(s)
exec(compile(s, str(__file__), 'exec'), {'__file__': __file__, '__name__': '__main__'})
