"""Local InternVL/Qwen2 checkpoint, native BF16 autoregressive NPU inference.

No weights or architecture files are changed. The checkpoint's own vision,
projector, Qwen2 parameters, conversation template and dynamic tiles are used.
"""
import contextlib, datetime, hashlib, importlib, json, logging, os, pathlib, re
import sys, time, traceback, types

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
import torch
import torch_npu
from PIL import Image
from universal_eval.model.interface import IModel

logger = logging.getLogger('universal_eval.model.adapters.internvl_qwen2_family_910b')


def fingerprint(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


class InferenceDropPath(torch.nn.Identity):
    """DropPath is exactly identity during evaluation; reject training use."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob:
            raise RuntimeError('This adapter implements evaluation only')
        return x


def load_local_classes(directory):
    directory = pathlib.Path(directory).resolve()
    name = 'internvl_npu_local_' + hashlib.sha256(str(directory).encode()).hexdigest()[:16]
    if name + '.modeling_internvl_chat' in sys.modules:
        return sys.modules[name + '.modeling_internvl_chat']
    package = types.ModuleType(name)
    package.__path__ = [str(directory)]
    sys.modules[name] = package
    vision = types.ModuleType(name + '.modeling_intern_vit')
    vision.__package__ = name
    vision.__file__ = str(directory / 'modeling_intern_vit.py')
    vision.DropPath = InferenceDropPath
    source = (directory / 'modeling_intern_vit.py').read_text()
    drop_path_imports = ['from timm.models.layers import DropPath', 'from timm.layers import DropPath']
    sites = [line for line in drop_path_imports if source.count(line) == 1]
    if len(sites) != 1:
        raise RuntimeError('Unreviewed DropPath import')
    patches = {
        sites[0]: '# Exact eval-mode identity supplied by the adapter',
        'torch.linspace(0, config.drop_path_rate, config.num_hidden_layers)':
            'torch.linspace(0, config.drop_path_rate, config.num_hidden_layers, device="cpu")',
    }
    for old, new in patches.items():
        if source.count(old) != 1:
            raise RuntimeError('Unreviewed architecture source; expected compatibility site: ' + old)
        source = source.replace(old, new)
    sys.modules[vision.__name__] = vision
    exec(compile(source, vision.__file__, 'exec'), vision.__dict__)
    return importlib.import_module(name + '.modeling_internvl_chat')


def prepare_pixels(image, config):
    """Existing project dynamic-tile policy and ImageNet normalization."""
    from torchvision import transforms
    size = int(config.force_image_size or config.vision_config.image_size)
    lo, hi = config.min_dynamic_patch, config.max_dynamic_patch
    ratios = sorted({(i, j) for n in range(lo, hi + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if lo <= i * j <= hi}, key=lambda pair: pair[0] * pair[1])
    width, height = image.size
    ratio, distance = (1, 1), float('inf')
    for candidate in ratios:
        diff = abs(width / height - candidate[0] / candidate[1])
        if diff < distance or (diff == distance and width * height > .5 * size * size * candidate[0] * candidate[1]):
            ratio, distance = candidate, diff
    resized = image.resize((ratio[0] * size, ratio[1] * size))
    tiles = [resized.crop((x, y, x + size, y + size))
             for y in range(0, ratio[1] * size, size)
             for x in range(0, ratio[0] * size, size)]
    if config.use_thumbnail and len(tiles) > 1:
        tiles.append(image.resize((size, size)))
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        mean=[.485, .456, .406], std=[.229, .224, .225])])
    return torch.stack([transform(tile) for tile in tiles])


class InternVLQwen2Family910BAdapter(IModel):
    def __init__(self, npu_devices=None, max_new_tokens=4096,
                 evidence_dir='workspace/diagnostics/model_runs', architecture_sha256=None, **kwargs):
        self.devices = list(npu_devices or [0])
        if len(self.devices) != 1:
            raise ValueError('This reviewed implementation uses one NPU')
        self.device = torch.device('npu:' + str(self.devices[0]))
        self.max_new_tokens = int(max_new_tokens)
        self.architecture_sha256 = dict(architecture_sha256 or {})
        stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime('%Y%m%d_%H%M%S')
        self.evidence = pathlib.Path(evidence_dir).resolve() / ('InternVL_Qwen2_910b_' + stamp + '_' + str(os.getpid()))
        self.evidence.mkdir(parents=True, exist_ok=False)
        self.trace_path = self.evidence / 'sample_execution.jsonl'
        self.details, self.lineage = {}, {}
        self.calls, self.profile_done = 0, False

    @property
    def model_name(self):
        return getattr(self, '_model_name', 'unloaded')

    @property
    def model_type(self):
        return 'InternVLChatModel'

    @property
    def inference_framework(self):
        return 'PyTorch + torch_npu / Transformers; native BF16 greedy generation'

    def download(self, *args, **kwargs):
        raise RuntimeError('Downloads are forbidden; use complete existing local weights')

    def load(self, model_path, **kwargs):
        from safetensors import safe_open
        from transformers import AutoTokenizer
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        if not torch.npu.is_available():
            raise RuntimeError('Ascend NPU is unavailable')
        directory = pathlib.Path(model_path).resolve()
        required_sources = {'config.json', 'configuration_intern_vit.py', 'configuration_internvl_chat.py',
                            'modeling_intern_vit.py', 'modeling_internvl_chat.py', 'conversation.py'}
        if set(self.architecture_sha256) != required_sources:
            raise RuntimeError('Review hashes are required for every local architecture source')
        for name, digest in self.architecture_sha256.items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Reviewed local architecture changed: ' + name)
        self._model_name = directory.parent.name + '/' + directory.name
        raw = json.loads((directory / 'config.json').read_text())
        if raw['llm_config']['architectures'] != ['Qwen2ForCausalLM']:
            raise ValueError('This adapter is approved only for the reviewed Qwen2 architecture')
        if raw.get('quantization_config') or raw.get('use_backbone_lora') or raw.get('use_llm_lora'):
            raise ValueError('Quantized/LoRA variants require a separate reviewed adapter')
        module = load_local_classes(directory)
        config_module = importlib.import_module(module.__package__ + '.configuration_internvl_chat')
        config = config_module.InternVLChatConfig(**raw)
        config.llm_config._attn_implementation = 'sdpa'
        config.vision_config.use_flash_attn = False
        with torch.device('meta'):
            self.model = module.InternVLChatModel(config, use_flash_attn=False).to(dtype=torch.bfloat16)
        self.model.language_model.config._attn_implementation = 'sdpa'
        if any(getattr(m, 'use_flash_attn', False) for m in self.model.vision_model.modules()):
            raise RuntimeError('CUDA flash-attention must not be enabled')
        parameters = dict(self.model.named_parameters())
        index_path = directory / 'model.safetensors.index.json'
        if index_path.is_file():
            indexed = json.loads(index_path.read_text())['weight_map']
        elif (directory / 'model.safetensors').is_file():
            with safe_open(str(directory / 'model.safetensors'), framework='pt', device='cpu') as handle:
                indexed = {key: 'model.safetensors' for key in handle.keys()}
        else:
            raise FileNotFoundError('A complete existing safetensors checkpoint is required')
        if set(parameters) != set(indexed):
            raise RuntimeError('Checkpoint/model parameter keys differ')
        loaded, records = set(), []
        for shard in sorted(set(indexed.values())):
            if pathlib.Path(shard).name != shard:
                raise ValueError('Unexpected shard path')
            with safe_open(str(directory / shard), framework='pt', device='cpu') as handle:
                if set(handle.keys()) != {k for k, f in indexed.items() if f == shard}:
                    raise RuntimeError('Shard index differs from actual tensors')
                for name in handle.keys():
                    value = handle.get_tensor(name)
                    expected_parameter = parameters[name]
                    if list(value.shape) != list(expected_parameter.shape) or value.dtype != torch.bfloat16:
                        raise RuntimeError('Unreviewed shape/dtype: ' + name)
                    target = value.to(self.device).contiguous()
                    positions = sorted({0, value.numel() // 2, value.numel() - 1})
                    expected = value.reshape(-1)[positions]
                    actual = target.reshape(-1)[torch.tensor(positions, device=self.device)].cpu()
                    if not torch.equal(expected, actual):
                        raise RuntimeError('Device weight mismatch: ' + name)
                    parent, leaf = name.rsplit('.', 1)
                    setattr(self.model.get_submodule(parent), leaf, torch.nn.Parameter(target, requires_grad=False))
                    records.append({'source_file': shard, 'source_key': name, 'target_key': name,
                                    'shape': list(value.shape), 'source_dtype': str(value.dtype),
                                    'execution_dtype': str(target.dtype), 'device': str(target.device),
                                    'slice_positions': positions, 'source_slice': expected.float().tolist(),
                                    'device_slice': actual.float().tolist()})
                    loaded.add(name)
            logger.info('NPU loaded %s: %s/%s parameters', shard, len(loaded), len(parameters))
        if loaded != set(parameters):
            raise RuntimeError('Incomplete checkpoint load')
        self.model.language_model.model.rotary_emb = Qwen2RotaryEmbedding(config.llm_config, device=self.device)
        for name, value in list(self.model.named_buffers()):
            if value.is_meta:
                raise RuntimeError('Unmaterialized buffer: ' + name)
        if any(value.device.type != 'npu' for value in self.model.parameters()):
            raise RuntimeError('A model parameter is not on NPU')
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(str(directory), trust_remote_code=True, local_files_only=True)
        self.image_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
        self.eos_id = self.tokenizer.convert_tokens_to_ids(self.model.conv_template.sep.strip())
        self.pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.eos_id
        if self.image_id in (None, self.tokenizer.unk_token_id) or self.eos_id in (None, self.tokenizer.unk_token_id) or not isinstance(self.eos_id, int):
            raise RuntimeError('Required local tokenizer special tokens are missing')
        self.model.img_context_token_id = self.image_id
        self._active_ids, self._features, self._bridge_verified = None, None, False
        def observe_bridge(model, args, values):
            if self._bridge_verified or values.get('inputs_embeds') is None:
                return
            actual = values['inputs_embeds'][self._active_ids == self.image_id]
            expected = self._features.reshape(-1, self._features.shape[-1])
            if actual.shape != expected.shape or not torch.equal(actual, expected) or not bool(expected.abs().sum() > 0):
                raise RuntimeError('Image embeddings were not faithfully injected into the language model')
            self._bridge_verified = True
        self.model.language_model.register_forward_pre_hook(observe_bridge, with_kwargs=True)
        loading = self.evidence / 'weight_loading.json'
        loading.write_text(json.dumps(records, indent=2))
        source_hashes = {}
        snapshot = self.evidence / 'source_snapshot'
        snapshot.mkdir()
        for name in ['config.json', 'configuration_intern_vit.py', 'configuration_internvl_chat.py',
                     'modeling_intern_vit.py', 'modeling_internvl_chat.py', 'conversation.py']:
            content = (directory / name).read_bytes()
            source_hashes[str(directory / name)] = hashlib.sha256(content).hexdigest()
            (snapshot / name).write_bytes(content)
        self.lineage = {
            'hardware': 'Ascend 910B', 'devices': self.devices, 'torch': torch.__version__,
            'torch_npu': torch_npu.__version__, 'precision': 'All original checkpoint parameters are BF16 on NPU; FP32 analytic rotary frequencies; no low-bit kernel, unpacking or substitute weights',
            'protocol': 'Native greedy autoregressive generation; checkpoint conversation template, dynamic tiles and original per-model configured generation budget; strict explicit MC extraction recorded separately',
            'weight_tensor_count': len(records), 'all_expected_parameters_loaded': True,
            'weight_loading_evidence': str(loading), 'per_sample_evidence': str(self.trace_path),
            'profiler_trace': str(self.evidence / 'npu_trace.json'),
            'architecture_source_sha256': source_hashes, 'architecture_source_snapshot': str(snapshot),
            'local_compatibility': ['DropPath is exact identity in evaluation; training use rejected',
                                    'DropPath schedule is constructed on CPU to permit meta parameter allocation',
                                    'Own chat/batch_chat CUDA device transfer methods are bypassed; same feature extraction/template and explicit native NPU transfers are used'],
            'vision_weight_source': 'Vision tower and projector loaded completely from this same checkpoint',
            'image_token_count_per_tile': self.model.num_image_token,
            'vision_attention': 'Checkpoint naive attention on NPU', 'language_attention': 'PyTorch SDPA on NPU',
            'module_placements': {n: str(next(m.parameters()).device) for n, m in self.model.named_modules()
                                  if n and any(True for _ in m.parameters(recurse=False))},
        }

    def _query(self, prompt, tile_count):
        template = self.model.conv_template.copy()
        template.system_message = self.model.system_message
        template.append_message(template.roles[0], '<image>\n' + prompt)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = '<img>' + '<IMG_CONTEXT>' * (self.model.num_image_token * tile_count) + '</img>'
        if query.count('<image>') != 1:
            raise RuntimeError('Expected one image placeholder')
        return query.replace('<image>', image_tokens, 1)

    def predict(self, prompt, media_path=None):
        self.calls += 1
        started = time.monotonic()
        entry = {'attempt': self.calls, 'prompt': prompt, 'media_path': media_path, 'error': None}
        self._bridge_verified = False
        try:
            media = pathlib.Path(media_path)
            with Image.open(media) as image:
                pixels = prepare_pixels(image.convert('RGB'), self.model.config)
            query = self._query(prompt, len(pixels))
            tokenized = self.tokenizer(query, return_tensors='pt')
            ids = tokenized['input_ids'].to(self.device)
            mask = tokenized['attention_mask'].to(self.device)
            self._active_ids = ids
            if int((ids == self.image_id).sum()) != len(pixels) * self.model.num_image_token:
                raise RuntimeError('Image-token count differs from projected features')
            def visual_prefill():
                self._features = self.model.extract_feature(pixels.to(device=self.device, dtype=torch.bfloat16))
                if not bool(torch.isfinite(self._features).all()):
                    raise RuntimeError('Nonfinite vision features')
                embeds = self.model.language_model.get_input_embeddings()(ids)
                embeds[ids == self.image_id] = self._features.reshape(-1, embeds.shape[-1])
                logits = self.model.language_model(inputs_embeds=embeds, attention_mask=mask,
                                                   use_cache=False, return_dict=True, logits_to_keep=1).logits
                if not bool(torch.isfinite(logits).all()):
                    raise RuntimeError('Nonfinite prefill logits')
                return embeds
            with torch.inference_mode():
                if not self.profile_done:
                    with contextlib.chdir(self.evidence):
                        with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                                  torch_npu.profiler.ProfilerActivity.NPU], record_shapes=True) as profile:
                            embeds = visual_prefill()
                            torch.npu.synchronize()
                        profile.export_chrome_trace(str(self.evidence / 'npu_trace.json'))
                    self.profile_done = True
                else:
                    embeds = visual_prefill()
                result = self.model.language_model.generate(
                    inputs_embeds=embeds, attention_mask=mask, max_new_tokens=self.max_new_tokens,
                    do_sample=False, eos_token_id=self.eos_id, pad_token_id=self.pad_id,
                    use_cache=True, return_dict_in_generate=True, output_scores=False)
            tokens = result.sequences[0].cpu().tolist()
            if len(tokens) > self.max_new_tokens:
                raise RuntimeError('Unexpected generation sequence including the prompt')
            raw = self.tokenizer.decode(tokens, skip_special_tokens=True).strip()
            truncated = len(tokens) >= self.max_new_tokens and (not tokens or tokens[-1] != self.eos_id)
            entry.update(source_image_sha256=hashlib.sha256(media.read_bytes()).hexdigest(),
                         pixel_sha256=fingerprint(pixels), pixel_shape=list(pixels.shape),
                         image_features_sha256=fingerprint(self._features), image_features_shape=list(self._features.shape),
                         input_ids=ids[0].cpu().tolist(), query=query, generation_token_ids=tokens,
                         generation_count=len(tokens), raw_output=raw, truncated=truncated,
                         stop_reason='length' if truncated else 'eos' if tokens and tokens[-1] == self.eos_id else 'other',
                         visual_embedding_injection_verified=self._bridge_verified)
            if not self._bridge_verified:
                raise RuntimeError('Actual image-to-language input was not observed')
            logger.info('InternVL NPU sample %s: tiles=%s generated=%s truncated=%s raw=%r',
                        self.calls, len(pixels), len(tokens), truncated, raw[:180])
            return raw
        except Exception:
            entry['error'] = traceback.format_exc()
            logger.error(entry['error'])
            raise
        finally:
            entry['elapsed_seconds'] = time.monotonic() - started
            clean = re.sub(r'\n+Answer the question by providing only the option letter.*$', '', prompt or '', flags=re.S).strip()
            self.details[(clean, media_path)] = entry
            with self.trace_path.open('a', encoding='utf8') as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
            self._features = None

    def batch_predict(self, batch):
        return [self.predict(prompt, media) for prompt, media in batch]

    def get_sample_execution(self, prompt, media_path):
        return self.details.get((prompt.strip(), media_path), {'error': 'No recorded execution'})

    def get_execution_lineage(self):
        return self.lineage

    def unload(self):
        self.model = None
        torch.npu.empty_cache()
