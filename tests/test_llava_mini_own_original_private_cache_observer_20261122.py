"""No learned forward: original cache body, graph metadata and full value capture."""
import ast,inspect,textwrap,unittest
import torch
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicLayer
from universal_eval.model.adapters import high_nine_llava_reference as original
from universal_eval.model.adapters.llava_mini_own_original_FP16_2048_native_v1_20261122 import OwnFourDimensionalDynamicLayer,OwnPrivateCacheLlama,OwnLlavaMiniOriginalFP16

class OriginalPrivateCacheAndCompleteObserverTests(unittest.TestCase):
    def test_original_cache_body_exact_and_shared_library_unchanged(self):
        source=inspect.getsource(DynamicLayer.lazy_initialization)
        tree=ast.parse(textwrap.dedent(inspect.getsource(original.LlavaMiniCUDAAdapter._ensure_cache_patch)))
        saved=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_lazy')
        private=ast.parse(textwrap.dedent(inspect.getsource(OwnFourDimensionalDynamicLayer.lazy_initialization))).body[0]
        self.assertEqual([ast.dump(n,include_attributes=False) for n in saved.body],[ast.dump(n,include_attributes=False) for n in private.body])
        layer=OwnFourDimensionalDynamicLayer();key=torch.arange(24,dtype=torch.float16).reshape(1,2,3,4);value=key+1
        layer.lazy_initialization(key,value)
        self.assertEqual(layer.keys.shape,torch.Size([1,2,0,4]));self.assertEqual(layer.values.shape,torch.Size([1,2,0,4]))
        self.assertEqual(inspect.getsource(DynamicLayer.lazy_initialization),source)
    def test_original_learned_graph_and_splice_functions_inherited(self):
        for name in ['_build_prompt_text','_tokenizer_image_token','_rotary','_encode_images_mini','_splice']:
            self.assertIs(getattr(OwnLlavaMiniOriginalFP16,name),getattr(original.LlavaMiniCUDAAdapter,name))
        self.assertIs(OwnPrivateCacheLlama.forward,LlamaForCausalLM.forward)
        self.assertIs(OwnPrivateCacheLlama.prepare_inputs_for_generation,LlamaForCausalLM.prepare_inputs_for_generation)
    def test_complete_raw_values_share_only_equal_shape_dtype_and_bits(self):
        observer=object.__new__(OwnLlavaMiniOriginalFP16);observer.arrays={};observer.occurrences=0
        a=observer._tensor(torch.ones((2,2),dtype=torch.float16));b=observer._tensor(torch.ones((2,2),dtype=torch.float16));c=observer._tensor(torch.ones(4,dtype=torch.float16));d=observer._tensor(torch.ones((2,2),dtype=torch.float32))
        self.assertEqual(a['array'],b['array']);self.assertNotEqual(a['array'],c['array']);self.assertNotEqual(a['array'],d['array']);self.assertEqual(len(observer.arrays),3);self.assertEqual(observer.occurrences,4)

if __name__=='__main__':unittest.main()
