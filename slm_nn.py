from slm_base import LanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers

# Too chatty!
transformers.logging.set_verbosity_error()
class NnLanguageModel(LanguageModel):
    def __init__(self):
        model_path = "EleutherAI/pythia-14m"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
    
    def untokenize(self, tokens):
        return self.tokenizer.decode(tokens)
   
    def tokenize(self, text):
        tokens = self.tokenizer(tokens, add_special_tokens=False, return_tensors='pt')
        return tokens['input_ids']

    def generate(self, tokens=None, max_tokens=9999999, do_sample=True):
        if isinstance(tokens, str):
            tokens = self.tokenizer(tokens, add_special_tokens=False, return_tensors='pt')
            tokens = tokens['input_ids']
        
        output = self.model.generate(tokens, do_sample=do_sample, max_new_tokens=max_tokens)[0]
        return output

    def generate_text(self, *args, **kwargs):
        return self.untokenize(self.generate(*args, **kwargs))


