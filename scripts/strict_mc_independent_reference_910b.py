"""Independent explicit-MC postprocessing reference; never credits ordinary articles."""
import re
def independent_choice(raw):
 s=str(raw).strip()
 for rule in [r'<([A-E])>',r'\[([A-E])\]',r'\(([A-E])\)',r'\*\*([A-E])\*\*',r'([A-E])']:
  m=re.fullmatch(rule,s,re.I)
  if m:return m[1].upper()
 m=re.fullmatch(r'<([A-E])>\s*([A-E])\s*</([A-E])>',s,re.I)
 if m and len({c.upper() for c in m.groups()})==1:return m[1].upper()
 for rule in [r'<answer>\s*([A-E])\s*</answer>',r'\\boxed\{\s*([A-E])\s*\}']:
  a=re.findall(rule,s,re.I)
  if a:return a[-1].upper()
 prefixes=[r'(?i:(?:the\s+)?(?:correct\s+)?answer\s*(?:is\s*:?|:)\s*)',r'(?i:(?:correct\s+)?option\s+(?:is\s+)?\s*)',r'(?i:(?:choose|select|pick)\s+)']
 for i,prefix in enumerate(prefixes):
  a=re.findall(prefix+r'[\'"(\[]\s*([A-Ea-e])\s*\.?\s*[\'"\])]',s)
  if a:return a[-1].upper()
  a=re.findall(prefix+(r'([A-Ea-e])' if i==1 else r'([A-E])')+r'(?=[\s.,:;!?]|$)',s)
  if a:return a[-1].upper()
 m=re.match(r'^([A-Ea-e])\s*[.)](?=\s|$)',s)
 if m:return m[1].upper()
 a=re.findall(r'(?m)^\s*([A-Ea-e])\s*$',s);return a[-1].upper() if a else None
assert independent_choice("a sample") is None and independent_choice("A sample of an object.") is None
