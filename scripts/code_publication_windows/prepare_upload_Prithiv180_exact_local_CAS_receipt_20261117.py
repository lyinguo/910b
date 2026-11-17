import sys
sys.dont_write_bytecode=True
from pathlib import Path
import ast
P=Path(__file__).resolve().parent
s=(P/'prepare_upload_Coder176_exact_local_receipt_20261117.py').read_text().replace('CoderGemma4E4B_local_result_sync_verified176_20261117','PrithivGemma4E4B_local_result_sync_verified180_20261117').replace('CoderGemma4E4B_original_HF_BF16_verified176_blank_G_publication_20261117','PrithivGemma4E4B_original_HF_BF16_verified180_blank_G_publication_20261117').replace('G212','G572').replace('upload_Coder176','upload_Prithiv180')
ast.parse(s);exec(compile(s,str(__file__),'exec'),{'__file__':__file__,'__name__':'__main__'})
