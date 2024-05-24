from slm_base import LanguageModel
from slm_torch import HuggingfaceTokenizer
import transformers
# Too chatty!
transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset, load_dataset
import datasets
datasets.utils.disable_progress_bar()

import torch.nn.functional as F
import torch

import datetime
from pathlib import Path
from glob import glob

class NnLanguageModel(LanguageModel):
    def __init__(self, model_path="EleutherAI/pythia-14m", model_name=None):
        self.model_path = model_path
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

    def predictor(self, context, do_sample=True):
        logits = self.forward(context)
        
        if do_sample:
            probabilities = F.softmax(logits, dim=-1)
            hit = torch.multinomial(probabilities, 1)[0]
        else:
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

    def get_trainer(self, dataset=None, num_train_epochs=1, output_dir='outputs', verbose=False):
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
            output_dir=output_dir,
            # Seems to be impossible to force the training
            # to a specific device
            use_cpu=(device.type == "cpu"),
            disable_tqdm=not verbose,
        )

        trainer = SFTTrainer(
            self.model,
            args=args,
            train_dataset=dataset,
            dataset_text_field='text',
            max_seq_length=512,
            tokenizer=self.tok_,
        )

        return trainer

    def train_dataset(self, *args, **kwargs):
        self.get_trainer().train()
    
    def generate(self, context, max_tokens=9999999, include_initial=True, end_token=None, pad_initial=True, do_sample=True):
        context = list(context)
        
        if include_initial:
            yield from context
        
        for i in range(max_tokens):
            if i >= max_tokens:
                return
            
            next_token = self.predictor(context, do_sample=do_sample)
            
            if next_token is None: return
            yield next_token
            if next_token == end_token: return
            context.append(next_token)

def get_latest_model(model_dir=None):
    from pathlib import Path

    if model_dir is None:
        return NnLanguageModel()
    
    model_name = "slm-"+Path(model_dir).name
    models = glob(f"{model_dir}/model-*.checkpoint")
    
    if not models:
        return NnLanguageModel(model_name=model_name)
    
    latest = sorted(models)[-1]

    return NnLanguageModel(latest, model_name=model_name)

def train(model_dir :str, *input_files :str, n_epochs :int=1):
    assert input_files, "Plz give some input files as parameters"
    dataset = load_dataset("text", data_files=input_files, sample_by='document', split="train")

    model = get_latest_model(model_dir)
    newpath = Path(model_dir)/("model-"+datetime.datetime.now().isoformat()+".checkpoint")
    trainer = model.get_trainer(dataset, verbose=True)
    trainer.train()
    trainer.save_model(newpath)

def blurb(model_dir :str, prompt :str, max_tokens:int =100):
    model = get_latest_model(model_dir)
    prompt = model.tokenize(prompt)
    output = model.generate(prompt, max_tokens=max_tokens)
    print(model.detokenize(output))

def push(model_dir :str):
    model = get_latest_model(model_dir)
    model.get_trainer(output_dir=model_dir).push_to_hub()

if __name__ == '__main__':
    import defopt
    defopt.run([train, blurb, push])

