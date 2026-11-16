"""HF renamed its own fast Qwen2 class; formatting and understanding mathematics are inherited unchanged."""
from transformers import Qwen2TokenizerFast
from .understanding import OwnOvisUnderstanding
from .configuration_ovis_u1 import Qwen3ConversationFormatter
class _OwnFormatterAlias(Qwen3ConversationFormatter):
 support_tokenizer_types=Qwen3ConversationFormatter.support_tokenizer_types+[Qwen2TokenizerFast.__name__]
class OwnOvisUnderstandingV2(OwnOvisUnderstanding):
 def get_conversation_formatter(self):
  assert isinstance(self.text_tokenizer,Qwen2TokenizerFast) and self.text_tokenizer.is_fast
  if getattr(self,'conversation_formatter',None) is None:self.conversation_formatter=_OwnFormatterAlias(self.text_tokenizer)
  return self.conversation_formatter
