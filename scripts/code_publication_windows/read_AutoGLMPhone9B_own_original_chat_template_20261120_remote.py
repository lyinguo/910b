import sys
sys.dont_write_bytecode=True
from pathlib import Path
import json,hashlib
p=Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/Glm4vForConditionalGeneration/zai-org/AutoGLM-Phone-9B/chat_template.jinja');b=p.read_bytes();print(json.dumps({'own_original_template':b.decode(),'SHA256':hashlib.sha256(b).hexdigest()}))
