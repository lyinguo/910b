"""Recognize explicit choice labels, including complete tagged labels, without articles."""
import re
from universal_eval.algorithm.inference import GreedyInference

def explicit_choice(raw):
    if not isinstance(raw,str):return None
    raw=raw.strip()
    # These complete responses unambiguously name one option.
    for rule in [r'<([A-E])>',r'\[([A-E])\]',r'\(([A-E])\)',r'\*\*([A-E])\*\*',r'([A-E])']:
        m=re.fullmatch(rule,raw,flags=re.I)
        if m:return m[1].upper()
    m=re.fullmatch(r'<([A-E])>\s*([A-E])\s*</([A-E])>',raw,flags=re.I)
    if m and len({x.upper() for x in m.groups()})==1:return m[1].upper()
    for rule in [r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}']:
        found=re.findall(rule,raw,flags=re.I)
        if found:return found[-1].upper()
    prefixes=[r'(?i:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*)',
              r'(?i:(?:correct\s+)?option\s+(?:is\s+)?\s*)',
              r'(?i:(?:choose|select|pick)\s+)']
    for index,prefix in enumerate(prefixes):
        # A quoted one-letter label can be lower-case. Close its delimiter so
        # "answer is 'a sample'" does not accidentally name option A.
        quoted=re.findall(prefix+r'[\'"(\[]\s*([A-Ea-e])\s*\.?\s*[\'"\])]',raw)
        if quoted:return quoted[-1].upper()
        letter=r'([A-Ea-e])' if index==1 else r'([A-E])'
        unquoted=re.findall(prefix+letter+r'(?=[\s.,:;!?]|$)',raw)
        if unquoted:return unquoted[-1].upper()
    leading=re.match(r'^([A-Ea-e])\s*[.)](?=\s|$)',raw)
    if leading:return leading[1].upper()
    lines=re.findall(r'(?m)^\s*([A-Ea-e])\s*$',raw)
    return lines[-1].upper() if lines else None

class NativeExplicitMCV2(GreedyInference):
    @property
    def algorithm_name(self):return 'native_explicit_mc_v2'
    def _extract_answer(self,raw_output,is_multiple_choice=False):
        return explicit_choice(raw_output) if is_multiple_choice else None
