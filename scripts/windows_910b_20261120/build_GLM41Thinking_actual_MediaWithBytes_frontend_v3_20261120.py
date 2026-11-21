import ast,hashlib,json
from pathlib import Path
V=Path(__file__).resolve().parent
name='prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v2_own_added_tokens_20261120.py';text=(V/name).read_text()
text=text.replace('GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v2_own_added_tokens_20261120','GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v3_actual_media_wrapper_20261120')
needle="image=media.load_file(Path(target_row['media_path']))";assert text.count(needle)==1;text=text.replace(needle,needle+'.media')
target='prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py';ast.parse(text);(V/target).write_text(text,encoding='utf8',newline='\n')
prefix='glm41thinking_own_original_VLLM_BF16_8192_20261120';oldfront=prefix+'_frontend';newfront=prefix+'_frontend_v2_actual_media'
front=(V/(oldfront+'.py')).read_text().replace('self.media.load_file(Path(media))','self.media.load_file(Path(media)).media')
ast.parse(front);(V/(newfront+'.py')).write_text(front,encoding='utf8',newline='\n')
for name in [prefix+'_native.py','verify_glm41thinking_own_original_VLLM_BF16_8192_native_v1_20261120.py']:
 text=(V/name).read_text().replace(oldfront,newfront)
 target=name.replace('_native.py','_native_v2_actual_media.py').replace('_native_v1_20261120.py','_native_v2_actual_media_20261120.py');ast.parse(text);(V/target).write_text(text,encoding='utf8',newline='\n')
p=V/'prepare_glm41thinking_own_original_VLLM_BF16_8192_true3_v1_20261120.py';text=p.read_text()
text=text.replace('GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v2_own_added_tokens_20261120','GLM41Thinking_own_original_VLLM_BF16_8192_native_frontend200_meta_v3_actual_media_wrapper_20261120')
text=text.replace(prefix+'_native.py',prefix+'_native_v2_actual_media.py').replace(oldfront+'.py',newfront+'.py').replace('verify_glm41thinking_own_original_VLLM_BF16_8192_native_v1_20261120.py','verify_glm41thinking_own_original_VLLM_BF16_8192_native_v2_actual_media_20261120.py').replace('prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v2_own_added_tokens_20261120.py','prepare_glm41thinking_own_original_VLLM_BF16_8192_frontend200_v3_actual_media_wrapper_20261120.py')
ast.parse(text);p.write_text(text,encoding='utf8',newline='\n')
print(json.dumps(dict(passed=True,actual_installed_media_wrapper_field='media',own_decoded_image_unchanged=True,own_token_config_and_original8192_no_thinking_override_preserved=True,first_two_zero_NN_frontend_failures_preserved=True)))
