from datasets import load_from_disk
from dp.phonemizer import Phonemizer
from speechbrain.inference.text import GraphemeToPhoneme
import cmudict
import re
import fire
import torch
from os.path import join

if torch.cuda.is_available() and torch.cuda.device_count() > 1:
    torch.multiprocessing.set_start_method('spawn')

class phonemization:
    def __init__(self):
        self.chars_to_ignore_regex = r'[,?.!-;:"–]'
        self.dp_phonemizer_model_path = join('models','d_phonemizer','en_us_cmudict_forward.pt')
        self.sb_phonemizer_model_path = join('models','soundchoice-g2p')

        
        self.cmu_dict = cmudict.dict()
        self.dp_phonemizer = Phonemizer.from_checkpoint(self.dp_phonemizer_model_path)
        if torch.cuda.is_available():
            self.sb_phonemizer = GraphemeToPhoneme.from_hparams(self.sb_phonemizer_model_path,run_opts={"device":"cuda"})
        else:
            self.sb_phonemizer = GraphemeToPhoneme.from_hparams(self.sb_phonemizer_model_path)
        self.normalize = False

    

        
        
    def dp_phonemize(self, text):
        return self.dp_phonemizer(text, lang='en_us',expand_acronyms=False).replace('[',' ').replace(']',' ').split()
    
    
    def cmu_phonemize(self, 
                      text, 
                      fallback_phonemizer=dp_phonemize):
        phoneme_lst=[]
        for word in text.split():
            if word in self.cmu_dict:
                phoneme_lst.extend(re.sub('[0-9]','',' '.join(self.cmu_dict.get(word)[0])).split())
            else:
                phoneme_lst.extend(fallback_phonemizer(self,word))
        phoneme_lst = [p.lower() for p in phoneme_lst]
        return(phoneme_lst)
    
    
    def sb_phonemize(self,text):
        return self.sb_phonemizer(text)

    def remove_special_characters(self,text):
        #print(text)
        return re.sub(self.chars_to_ignore_regex, ' ', text).lower() + " "

    def replace_multiple_spaces_with_single_space(self, input_string):
        """Replace multiple spaces with a single space."""
        return re.sub(r'\s+', ' ', input_string)
        
    def phonemize_batch(self, 
                        batch, 
                        phonamizer_fn=dp_phonemize, 
                        suffix=''):
        
        if self.normalize:
            text = batch['text_norm'].lower()
        else:
            text = batch['text'].lower()
        phoneme_str = ' '.join(phonamizer_fn(text))
        phoneme_str = phoneme_str.lower()
        phoneme_str = self.replace_multiple_spaces_with_single_space(phoneme_str)
        batch[f'phoneme{suffix}'] = phoneme_str.strip()
        return batch

    def remove_special_characters_batch(self, batch):
        batch["text_norm"] = self.remove_special_characters(batch["text"])
        return batch
        
    def run(self, 
            dataset_path, 
            output_path, 
            phonemizers=('dp','sb','cmu'), 
            normalize=True, 
            nproc=1):
       
        self.normalize = normalize
        data = load_from_disk(dataset_path)
        if isinstance(phonemizers, str):
            phonemizers = (phonemizers,)
        if normalize:
            data = data.map(self.remove_special_characters_batch, num_proc=nproc)
        for phonemizer in phonemizers:
            if phonemizer == 'cmu':
                print('cmu phonemization')
                data = data.map(self.phonemize_batch, fn_kwargs={'phonamizer_fn':self.cmu_phonemize,'suffix':'_cmu'},num_proc=nproc)                
            if phonemizer == 'dp':
                print('dp phonemization')
                data = data.map(self.phonemize_batch, fn_kwargs={'phonamizer_fn':self.dp_phonemize,'suffix':'_dp'},num_proc=nproc)
            if phonemizer == 'sb':
                print('sb phonemization')
                if torch.cuda.is_available():
                    nproc = torch.cuda.device_count()
                data = data.map(self.phonemize_batch, fn_kwargs={'phonamizer_fn':self.sb_phonemize,'suffix':'_sb'},num_proc=nproc, cache_file_name='/g/data/iv96/mostafa/cache_sb', load_from_cache_file=False)
        data.save_to_disk(output_path)


if __name__=='__main__':
    fire.Fire(phonemization)
    
