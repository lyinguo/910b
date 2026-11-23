"""Validate Kanana's own dedicated inputs; no learned model or NPU forward.

The reference implements this checkpoint's resize, normalization, patch order,
conversation and media-token insertion independently. All 200 raw inputs are
checked; no other model's frontend, weights or outputs are used.
"""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
from datetime import datetime, timezone
from decimal import Decimal
import hashlib, inspect, json, math, os, re, time, traceback, zipfile
import xml.etree.ElementTree as ET

R = Path('/root/projects/comperror')
D = R/'workspace/diagnostics/batch_910b'
B = D/'A100_relative5pct_alignment_20261103/request_20261103_user_A100_relative5pct'
H = B/'Kanana_own_original_BF16_dedicated_frontend200_v2_original_Jinja_whitespace_20261123'
MID = 'kakaocorp/kanana-1.5-v-3b-instruct'
W = Path('/mnt/afs/huggingfaceautoclonemodels/downmodels/KananaVForConditionalGeneration')/MID
PKG = 'universal_eval.model.kanana_own_original_graphtext_v2_20261123'
YAML = R/'configs/KananaVForConditionalGeneration/kakaocorp--kanana-1.5-v-3b-instruct.yaml'
SERVER = Path('/root/projects/C500_MLU_与A100差距5%内_同值运行已选定_20261029.xlsx')
ORIG = B/'local_workbook_delivery_20261103/local_A100_reference_snapshot.json'
DATA = R/'workspace/datasets/image-text-to-text-mmbench/eval.jsonl'
DATA_SHA = '2f5184c8d02695ef95ef16efe0c48640a0d99c680ab06f6847ec5d1372b06451'


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def put(name, value):
    path = H/name
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    os.replace(tmp, path)


def workbook_row(path):
    ns = {'x': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings = []
        if 'xl/sharedStrings.xml' in z.namelist():
            root = ET.fromstring(z.read('xl/sharedStrings.xml'))
            strings = [''.join(q.itertext()) for q in root]
        root = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
        matches = []
        for row in root.findall('x:sheetData/x:row', ns):
            cells = {}
            for cell in row:
                ref = cell.attrib.get('r', '')
                col = re.sub(r'\d+', '', ref)
                value = cell.find('x:v', ns)
                value = (value.text or '') if value is not None else ''
                if cell.attrib.get('t') == 's':
                    value = strings[int(value)]
                elif cell.attrib.get('t') == 'inlineStr':
                    value = ''.join(cell.find('x:is', ns).itertext())
                cells[col] = value
            if cells.get('C') == MID:
                matches.append(dict(row=int(row.attrib['r']), cells=cells))
    assert len(matches) == 1, matches
    match = matches[0]
    assert match['cells']['B'] == 'image-text-to-text'
    assert match['cells'].get('G', '') == ''
    return match


def independent_pixels(image, processor):
    import numpy as np
    from PIL import Image
    image = image.convert('RGB')
    width, height = image.size
    if min(width, height) < 28:
        scale = 28/min(width, height)
        image = image.resize((int(width*scale), int(height*scale)), Image.Resampling.BICUBIC)
    width, height = image.size
    rh, rw = round(height/28)*28, round(width/28)*28
    if rh*rw > processor.max_pixels:
        beta = math.sqrt(height*width/processor.max_pixels)
        rh, rw = math.floor(height/beta/28)*28, math.floor(width/beta/28)*28
    elif rh*rw < processor.min_pixels:
        beta = math.sqrt(processor.min_pixels/(height*width))
        rh, rw = math.ceil(height*beta/28)*28, math.ceil(width*beta/28)*28
    resized = np.asarray(image.resize((rw, rh), Image.Resampling.BICUBIC))
    # Original HF rescale: float64 multiply, then cast back to float32.
    values = (resized.astype(np.float64)*(1/255)).astype(np.float32)
    mean = np.asarray(processor.image_mean, dtype=np.float32)
    std = np.asarray(processor.image_std, dtype=np.float32)
    values = (values-mean)/std
    gh, gw = rh//14, rw//14
    frames = np.stack([values.transpose(2, 0, 1)]*2, axis=0)
    reference = frames.reshape(1, 2, 3, gh//2, 2, 14, gw//2, 2, 14)
    reference = reference.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    reference = np.ascontiguousarray(reference.reshape(gh*gw, 1176))
    meta = dict(vision_grid_thw=(1, gh, gw), hw_best_resolution=(rh, rw),
                hw_orig_resolution=(height, width), image_token_thw=(1, gh//2, gw//2))
    return reference, meta, resized


INTRO = '''The following is a conversation between a curious human and AI assistant. 당신은 Kakao에서 개발된 인공지능 언어모델이고 이름은 kanana입니다.
Knowledge Cutoff Date: June30, 2024.
Capabilities and Limitations:
   - I cannot search for external content such as weather, news, or the current date and time.
   - If a URL is provided, I cannot access it directly. Insteaed, please copy and provide the relevant content for me to process.'''


def independent_ids(prompt, image_tokens_hw, backend, bos):
    text = bos+INTRO+'\nHuman: <image>\nHuman: '+prompt+'\n\nAI: '
    h, w = image_tokens_hw
    ids = []
    # The original wrapper intentionally tokenizes role boundaries separately.
    chunks = re.split(r'(<image>|\nAI: |\nHuman: )', text)
    newline = backend.encode('\n', add_special_tokens=False).ids
    for chunk in chunks:
        if chunk == '<image>':
            ids += newline
            for row in range(h):
                ids += [-1]*w
                if row != h-1:
                    ids += newline
        elif chunk:
            ids += backend.encode(chunk, add_special_tokens=False).ids
    return text, ids


def clean(text):
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    return re.sub(r'<image\d+>\s*', '', text)


def main():
    H.mkdir(exist_ok=False)
    start = time.monotonic()
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    sys.path[:0] = [str(R), str(R/'scripts')]
    import importlib
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    from tokenizers import Tokenizer
    from universal_eval.algorithm.inference import GreedyInference
    torch.set_num_threads(4)
    plan = json.loads(Path(sys.argv[1]).read_bytes())
    ptr = json.loads((D/'A100_ALIGNMENT_HANDOFF_CURRENT.json').read_bytes())
    canonical_before = {p: sha(p) for p in ptr['canonical_SHA256']}
    assert canonical_before == ptr['canonical_SHA256']
    assert sha(YAML) == '4eb9bb0dc87e8fd6ff212e1cc7fa7e0ccf8c5591fa878eab476cac2809cc5121'
    saved = yaml.safe_load(YAML.read_text())
    assert saved['model']['name'] == MID and saved['model']['kwargs']['max_new_tokens'] == 2048
    assert saved['algorithm']['batch_size'] == 1 and saved['model']['kwargs']['torch_dtype'] == 'auto'
    assert sha(SERVER) == plan['local_book']['SHA256'] == '2981b33d68e4825843b3ead4b21a9a53b1931232be497cf5751373e33ead8878'
    server = workbook_row(SERVER)
    assert server == plan['local_row']
    assert sha(ORIG) == '700956054f482a3c6c81ae55de1ae36435ba9e0e15ae81450848af081f0995ba'
    original = json.loads(ORIG.read_bytes())['models'][MID]
    values = dict(server=server['cells']['D'], local=plan['local_row']['cells']['D'], preserved_original=str(original['A100_D']))
    assert all(Decimal(z).is_finite() and Decimal(z)>0 for z in values.values())
    assert sha(DATA) == DATA_SHA
    original_path = Path(saved['data']['dataset_path'])
    if original_path.is_file():
        assert sha(original_path) == DATA_SHA
    data_provenance = dict(saved_path=str(original_path), saved_path_currently_readable=original_path.is_file(),
                           actual_own_project_dataset=str(DATA), frozen_shared_original_dataset_SHA256=DATA_SHA)
    source_hashes = {}
    for name in ['processing.py', 'processing_image.py', 'tokenization.py', 'configuration.py']:
        own = W/name
        local = R/('universal_eval/model/kanana_own_original_graphtext_v2_20261123/'+name)
        assert own.read_bytes() == local.read_bytes()
        source_hashes[str(own)] = sha(own)
        source_hashes[str(local)] = sha(local)
    assets = {str(W/name): sha(W/name) for name in ['config.json','tokenizer.json','tokenizer_config.json','preprocessor_config.json','generation_config.json','model.safetensors.index.json']}
    if (W/'special_tokens_map.json').is_file():
        assets[str(W/'special_tokens_map.json')] = sha(W/'special_tokens_map.json')
    modt = importlib.import_module(PKG+'.tokenization')
    modi = importlib.import_module(PKG+'.processing_image')
    modp = importlib.import_module(PKG+'.processing')
    tokenizer = modt.KananaVTokenizerFast.from_pretrained(str(W), local_files_only=True)
    before_probe = tokenizer.encode('The quick brown fox jumps over the lazy dog.', add_special_tokens=False)
    raw = json.loads((W/'tokenizer.json').read_bytes())
    merges = raw['model'].get('merges', [])
    if merges and isinstance(merges[0], list):
        raw['model']['merges'] = [' '.join(x) if isinstance(x, list) else x for x in merges]
    backend = Tokenizer.from_str(json.dumps(raw))
    tokenizer._tokenizer = backend
    independent_backend = Tokenizer.from_str(json.dumps(raw))
    assert tokenizer.eos_token_id == 128009 and tokenizer.pad_token_id == 128001
    image_processor = modi.KananaVImageProcessor(**json.loads((W/'preprocessor_config.json').read_bytes()))
    processor = modp.KananaVProcessor(image_processor, tokenizer)
    assert image_processor.min_pixels == 78400 and image_processor.max_pixels == 1254400
    assert image_processor.patch_size == 14 and image_processor.temporal_patch_size == image_processor.merge_size == 2
    put('native_processor_and_original_generation_contract.json', dict(tokenizer_class=type(tokenizer).__name__,
        actual_eos_token_id=tokenizer.eos_token_id,actual_pad_token_id=tokenizer.pad_token_id,
        before_canonical_merges_probe=before_probe,after_canonical_merges_probe=tokenizer.encode('The quick brown fox jumps over the lazy dog.',add_special_tokens=False),
        original_generate=dict(max_new_tokens=2048,do_sample=False,temperature=0.,eos_token_id=128009,pad_token_id=128001),
        original_dedicated_input_max_length=8192,original_batch_size=1,current_C_unique=server,
        current_three_A100_D=values,original_snapshot_record=original,dataset_provenance=data_provenance))
    # Save exact installed transforms as external evidence, never model source_sha.
    from transformers import image_transforms
    library = dict(rescale=inspect.getsource(image_transforms.rescale),normalize=inspect.getsource(image_transforms.normalize),resize=inspect.getsource(image_transforms.resize))
    put('installed_original_HF_image_transform_external_sources.json',library)
    rows = [json.loads(line) for line in DATA.read_text().splitlines()]
    assert len(rows) == 4329
    records = []
    with (H/'own_original_frontend200.jsonl').open('x', encoding='utf8') as out:
        for index, row in enumerate(rows[:200]):
            path = Path(row['media_path'])
            path = path if path.is_absolute() else DATA.parent/path
            path = path.resolve()
            assert path.is_file()
            image = Image.open(path).convert('RGB')
            prompt = clean(row['prompt'])
            model_prompt = GreedyInference._add_answer_instruction(prompt)
            conv = [dict(role='user',content=[dict(type='image'),dict(type='text',text=model_prompt)])]
            actual = processor(dict(conv=conv,image=[image]),max_length=8192,add_generation_prompt=True)
            reference, meta, resized = independent_pixels(image, image_processor)
            pixel = actual['image'].contiguous().numpy()
            assert pixel.dtype == reference.dtype == np.dtype('float32')
            assert pixel.shape == reference.shape and np.array_equal(pixel.view(np.uint32),reference.view(np.uint32)), (index,'all pixel bits differ',float(np.max(np.abs(pixel-reference))))
            assert actual['image_meta'] == {key:[value] for key,value in meta.items()}
            text, ids = independent_ids(model_prompt,meta['image_token_thw'][1:],independent_backend,tokenizer.bos_token)
            if text != actual['text_raw']:
                put('prompt_reference_first_difference.json',dict(index=index,actual=actual['text_raw'],independent=text))
            assert text == actual['text_raw'],(index,'original conversation text differs')
            ai = actual['text']['input_ids'].numpy()
            assert ai.tolist() == ids,(index,'own full token IDs differ')
            assert torch.equal(actual['text']['attention_mask'],torch.ones_like(actual['text']['input_ids']))
            assert int((ai == -1).sum()) == meta['image_token_thw'][1]*meta['image_token_thw'][2]
            if index in (0,99,199):
                np.savez_compressed(H/f'necessary_original_input_example_{index:05d}.npz',resized_RGB_uint8=resized,
                    actual_original_F32_pixels=pixel,independent_F32_pixels=reference,full_own_input_ids=ai,
                    full_attention_mask=actual['text']['attention_mask'].numpy(),full_grid_thw=np.asarray(meta['vision_grid_thw'],dtype=np.int64))
            rec=dict(index=index,prompt=prompt,reference=row['reference'],media_path=str(path),image_SHA256=sha(path),
                     original_greedy_model_prompt=model_prompt,original_dedicated_conversation=text,
                     input_ids=ids,attention_mask=actual['text']['attention_mask'].tolist(),input_tokens=len(ids),
                     input_ids_SHA256=hashlib.sha256(ai.tobytes()).hexdigest(),pixel_shape=list(pixel.shape),pixel_dtype='float32',
                     pixel_SHA256=hashlib.sha256(pixel.tobytes()).hexdigest(),image_meta=actual['image_meta'],
                     all_original_pixel_bits_and_input_tokens_independently_equal=True,no_model_forward=True)
            out.write(json.dumps(rec,ensure_ascii=False)+'\n');out.flush();records.append(rec)
            put('state.json',dict(UTC=datetime.now(timezone.utc).isoformat(),phase='own_original_dedicated_CPU_frontend_only',samples=len(records),target=200,no_NN_or_fullNF=True))
    assert {p:sha(p) for p in canonical_before} == canonical_before
    review=dict(passed=True,UTC=datetime.now(timezone.utc).isoformat(),model_id=MID,samples=len(records),
                source_SHA256=source_hashes,external_assets_SHA256=assets,frontend=str(H/'own_original_frontend200.jsonl'),
                frontend_SHA256=sha(H/'own_original_frontend200.jsonl'),dataset_provenance=data_provenance,
                current_C_unique_row=server['row'],current_three_A100_D=values,
                all200_original_F32_pixel_bits_and_tokens_independently_equal=True,
                original_EOS128009_pad128001_max2048_outer1_and_input8192_preserved=True,
                learned_CPU_GPU_NPU_forwards=0,formal_NEW_NPU3=False,formal_fullNF_or_200_or_qualification=False,
                all_checkpoint_values_untouched=True,no_G_or_canonical_write=True,elapsed_seconds=time.monotonic()-start)
    put('review.json',review)
    put('actual_completion.json',dict(UTC=review['UTC'],actual_returncode=0,no_NN_or_fullNF=True,frontend_only=True))
    print(json.dumps(dict(passed=True,home=str(H),review_SHA256=sha(H/'review.json'),samples=200,frontend_SHA256=review['frontend_SHA256'],
                         current_C_unique_row=server['row'],current_three_A100_D=values,no_learned_model_or_fullNF=True),ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        if H.is_dir() and not (H/'actual_completion.json').exists():
            put('actual_completion.json',dict(UTC=datetime.now(timezone.utc).isoformat(),actual_returncode=1,error=traceback.format_exc(),no_NN_or_fullNF=True))
        raise
