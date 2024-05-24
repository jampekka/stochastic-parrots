from slm_base import LanguageModel
from slm_torch import HuggingfaceTokenizer
import transformers
# Too chatty!
#transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset
import datasets
datasets.utils.disable_progress_bar()

import torch.nn.functional as F

import torch


class NnLanguageModel(LanguageModel):
    def __init__(self):
        model_path = "EleutherAI/pythia-14m"
        self.tok_ = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer = HuggingfaceTokenizer(self.tok_)

        # Hacking these because something eats the original
        # special tokens
        """
        self.custom_end_string = '$\\'
        self.custom_end_token_id = self.tokenizer.convert_tokens_to_ids('$\\')
        self.custom_start_string = ''
        """

        self.tok_.add_special_tokens({'pad_token': '<|padding|>'})
        #self.tokenizer.pad_token = self.tokenizer.eos_token
        #print(self.tokenizer.pad_token_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self._trainer = None
    

    def untokenize(self, tokens):
        return self.tokenizer.decode(tokens)
   
    def tokenize(self, text):
        tokens = self.tok_(text, return_tensors='pt')
        return tokens['input_ids'][0]
    
    """
    def generate(self, tokens=None, max_tokens=9999999, do_sample=True, **kwargs):
        # The end and start handlings are a total mess!
        if isinstance(tokens, str):
            tokens = self.tokenizer(tokens, add_special_tokens=True, return_tensors='pt')
        else:
            tokens = {'input_ids': tokens}
        
        output = self.model.generate(**tokens,
                    do_sample=do_sample,
                    max_new_tokens=max_tokens,
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=[self.custom_end_token_id, self.tokenizer.eos_token_id],
                    **kwargs)
        return output[0][len(tokens['input_ids']):]

    def generate_text(self, *args, **kwargs):
        # Really struggling with end tokens here!
        out = self.untokenize(self.generate(*args, **kwargs))
        # Horrible hackery!
        return out
    """
    
    def forward(self, context):
        if isinstance(context, str):
            context = self.tokenize(context)
        context = torch.as_tensor(context)
        out = self.model.forward(context.reshape(1, -1))[0][0][-1]
        return out

    def next_token_probs(self, context):
        logits = self.forward(context)
        return F.softmax(logits).detach()

    def predictor(self, context):
        logits = self.forward(context)
        
        
        hit = torch.argmax(logits)
        assert hit < len(logits)
        return hit

    def train_text(self, text, **kwargs):
        return self.train_texts([text], **kwargs)

    def train_texts(self, texts, **kwargs):
        # This is very expensive for on-the-fly data
        # TODO: Figure out the Dataset
        def gen():
            for text in texts:
                #text = self.custom_start_string + text + self.custom_end_string + self.tokenizer.eos_token
                yield {'text': text}
        dataset = Dataset.from_generator(gen)
        return self.train_dataset(dataset, **kwargs)
    
    #def generate(self, context, max_tokens=9999999, include_initial=True, end_token=None, pad_initial=True):
    
    def train(self, tokens, **kwargs):
        text = self.detokenize(tokens)
        return self.train_text(text, **kwargs)

    def train_dataset(self, dataset, num_train_epochs=1):

        # Not foolproof, but torch device manager is a major PITA
        device = torch.get_default_device()
        args = transformers.TrainingArguments(
            auto_find_batch_size=True,
            #gradient_accumulation_steps=4,
            #warmup_steps=2,
            #max_steps=100,
            num_train_epochs=num_train_epochs,
            #learning_rate=2e-4,
            #evaluation_strategy = 'steps',
            #eval_accumulation_steps = 1, 
            #eval_steps = 10,
            seed = 42,
            #fp16=True,
            #logging_steps=1,
            output_dir='outputs',
            # Seems to be impossible to force the training
            # to a specific device
            use_cpu=(device.type == "cpu"),
            
            disable_tqdm=True,
        )

        trainer = SFTTrainer(
            self.model,
            args=args,
            train_dataset=dataset,
            dataset_text_field='text',
            max_seq_length=512,
            tokenizer=self.tok_,
        )

        trainer.train()
    
    def generate(self, context, max_tokens=9999999, include_initial=True, end_token=None, pad_initial=True):
        context = list(context)
        
        if include_initial:
            yield from context
        
        for i in range(max_tokens):
            if i >= max_tokens:
                return
            
            next_token = self.predictor(context)
            
            if next_token is None: return
            yield next_token
            if next_token == end_token: return
            context.append(next_token)

