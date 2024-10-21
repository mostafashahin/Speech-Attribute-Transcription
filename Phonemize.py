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
        self.normalize = False

    

    def split_by_noise(self, text):
        text_split = []
        
        # Find indices for noise boundaries
        ind_s = [i for i, item in enumerate(text) if self.noise_bound[0] in item]
        ind_e = [i for i, item in enumerate(text) if self.noise_bound[1] in item]
        
        # Ensure start and end markers match in length
        if not ind_s or len(ind_s) != len(ind_e):
            text_split.append(('T',text))
            return text_split
        
        # Add text before the first noise segment, if any
        if ind_s[0] > 0:
            pre_noise_text = text[:ind_s[0]].strip()
            if pre_noise_text:
                text_split.append(('T', pre_noise_text))
        
        # Process each noise segment and the text between noise segments
        for i in range(len(ind_s)):
            # Add the noise segment
            noise_segment = text[ind_s[i]:ind_e[i] + 1]
            text_split.append(('N', noise_segment))
            
            # Add the text between noise segments (or after the last one)
            if i < len(ind_s) - 1:
                between_noise_text = text[ind_e[i] + 1:ind_s[i + 1]].strip()
                if between_noise_text:
                    text_split.append(('T', between_noise_text))
            else:
                post_noise_text = text[ind_e[i] + 1:].strip()
                if post_noise_text:
                    text_split.append(('T', post_noise_text))
        
        return text_split

        
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
            text = batch['text_norm'].lower().strip()
        else:
            text = batch['text'].lower().strip()
        if text:
            
            if self.hand_noise:
                phonemes = []
                for tag, cont in self.split_by_noise(text):
                    if tag == 'N':
                        if not self.ignor_noise:
                            if self.noise_out_symb:
                                phonemes.append(self.noise_out_symb)
                            else:
                                phonemes.append(cont)
                    elif tag == 'T':
                        phonemes.extend(phonamizer_fn(cont))
                    else:
                        pass
            else:
                phonemes = phonamizer_fn(text)                     
            phoneme_str = ' '.join(phonemes)
            phoneme_str = phoneme_str.lower()
            phoneme_str = self.replace_multiple_spaces_with_single_space(phoneme_str)
        else:
            phoneme_str = ''
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
            noise_bound='<>',  # Should be exactly two characters, defining the boundaries of the noise tag (e.g., '<noise>')
            hand_noise=True,  # Determines whether or not to handle noise tags in the text; if True, noise tags are either retained in the same position or replaced by `noise_out_symb`. If False, noise tags are treated as regular text and passed to the phonemizer.
            noise_out_symb='<unk>',  # The symbol to use in place of noise tags when `hand_noise` is True.
            ignor_noise=False,  # If True, noise tags will be completely removed from the output, without replacement.
            nproc=1):  # The number of processes to use for parallelization, default is 1 (single process).
       
        self.normalize = normalize
        self.hand_noise = hand_noise 
        self.noise_bound = noise_bound
        self.noise_out_symb = noise_out_symb
        self.ignor_noise = ignor_noise
        
        data = load_from_disk(dataset_path)
        if isinstance(phonemizers, str):
            phonemizers = (phonemizers,)
        if normalize:
            data = data.map(self.remove_special_characters_batch, num_proc=nproc)
        for phonemizer in phonemizers:
            if phonemizer == 'cmu':
                self.cmu_dict = cmudict.dict()
                self.dp_phonemizer = Phonemizer.from_checkpoint(self.dp_phonemizer_model_path)
                print('cmu phonemization')
                data = data.map(self.phonemize_batch, fn_kwargs={'phonamizer_fn':self.cmu_phonemize,'suffix':'_cmu'},num_proc=nproc)                
            if phonemizer == 'dp':
                self.dp_phonemizer = Phonemizer.from_checkpoint(self.dp_phonemizer_model_path)
                print('dp phonemization')
                data = data.map(self.phonemize_batch, fn_kwargs={'phonamizer_fn':self.dp_phonemize,'suffix':'_dp'},num_proc=nproc)
            if phonemizer == 'sb':
                if torch.cuda.is_available():
                    self.sb_phonemizer = GraphemeToPhoneme.from_hparams(self.sb_phonemizer_model_path,run_opts={"device":"cuda"})
                else:
                    self.sb_phonemizer = GraphemeToPhoneme.from_hparams(self.sb_phonemizer_model_path)
                print('sb phonemization')
                if torch.cuda.is_available():
                    nproc = torch.cuda.device_count()
                data = data.map(self.phonemize_batch, fn_kwargs={'phonamizer_fn':self.sb_phonemize,'suffix':'_sb'},num_proc=nproc, load_from_cache_file=False)
        data.save_to_disk(output_path)


if __name__=='__main__':
    fire.Fire(phonemization)
    
