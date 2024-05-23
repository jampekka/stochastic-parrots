from slm_base import LanguageModel
import transformers
# Too chatty!
#transformers.logging.set_verbosity_error()

from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer
from datasets import Dataset
import datasets
datasets.utils.disable_progress_bar()

import torch


class NnLanguageModel(LanguageModel):
    def __init__(self):
        model_path = "EleutherAI/pythia-14m"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self._trainer = None
    

    def untokenize(self, tokens):
        return self.tokenizer.decode(tokens)
   
    def tokenize(self, text):
        tokens = self.tokenizer(tokens, add_special_tokens=False, return_tensors='pt')
        return tokens['input_ids']

    def generate(self, tokens=None, max_tokens=9999999, do_sample=True, **kwargs):
        if isinstance(tokens, str):
            tokens = self.tokenizer(tokens, add_special_tokens=False, return_tensors='pt')
        else:
            tokens = {'input_ids': tokens}
        
        output = self.model.generate(**tokens, do_sample=do_sample, max_new_tokens=max_tokens, **kwargs)[0]
        return output

    def generate_text(self, *args, **kwargs):
        return self.untokenize(self.generate(*args, **kwargs))

    def train_text(self, text, **kwargs):
        return self.train_texts([text], **kwargs)

    def train_texts(self, texts, **kwargs):
        # This is very expensive for on-the-fly data
        # TODO: Figure out the Dataset
        def gen():
            for text in texts:
                text = self.tokenizer.bos_token + text + self.tokenizer.eos_token
                print(text)
                yield {'text': text}
        dataset = Dataset.from_generator(gen)
        return self.train_dataset(dataset, **kwargs)

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
        )

        trainer.train()



        
        




