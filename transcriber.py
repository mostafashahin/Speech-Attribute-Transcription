import fire
import logging
import sys, os
import yaml
import json
import torch
import librosa
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2ForCTC
import transformers
import pandas as pd
from datasets import load_from_disk, DatasetDict
import evaluate
import re
from metrics import cm

logger = logging.getLogger(__name__)
# Setup logging
logger.setLevel(logging.ERROR)
console_handler = logging.StreamHandler()
formater = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",)
console_handler.setFormatter(formater)
console_handler.setLevel(logging.ERROR)

logger.addHandler(console_handler)


class transcribe_SA():
    def __init__(self, model_path, verbose=0):
        if verbose == 0:
            logger.setLevel(logging.ERROR)
            transformers.logging.set_verbosity_error()
            #console_handler.setLevel(logging.ERROR)
        elif verbose == 1:
            logger.setLevel(logging.WARNING)
            transformers.logging.set_verbosity_warning()
            #console_handler.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.INFO)
            transformers.logging.set_verbosity_info()
            #console_handler.setLevel(logging.INFO)
        # Read YAML file
        logger.info('Init Object')
        if torch.cuda.is_available():
            self.accelerate = True
            self.device = torch.device('cuda')
            self.n_devices = torch.cuda.device_count()
            assert self.n_devices == 1, 'Support only single GPU. Please use CUDA_VISIBLE_DEVICES=gpu_index if you have multiple gpus' #Currently support only single gpu
        else:
            self.device = torch.device('cpu')
            self.n_devices = 1
        self.model_path = model_path
        self.load_model()
        self.get_available_attributes()
        self.get_att_binary_group_indexs()

    def load_model(self):
        if not os.path.exists(self.model_path):
            logger.error(f'Model file {self.model_path} is not exist')
            raise FileNotFoundError

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.sampling_rate = self.processor.feature_extractor.sampling_rate

    def get_available_attributes(self):
        if not hasattr(self, 'model'):
            logger.error('model not loaded, call load_model first!')
            raise AttributeError("model not defined")
        att_list = set(self.processor.tokenizer.get_vocab().keys()) - set(self.processor.tokenizer.all_special_tokens)
        att_list = [p.replace('p_','') for p in att_list if p[0]=='p']
        self.att_list = att_list

    def print_availabel_attributes(self):
        print(self.att_list)

    
    def get_att_binary_group_indexs(self):
        self.group_ids = [] #Each group contains the token_ids of [<PAD>, n_att, p_att] sorted by their token ids
        for i, att in enumerate(self.att_list):
            n_indx = self.processor.tokenizer.convert_tokens_to_ids(f'n_{att}')
            p_indx = self.processor.tokenizer.convert_tokens_to_ids(f'p_{att}')
            self.group_ids.append(sorted([self.pad_token_id, n_indx, p_indx]))

    def decode_att(self, logits, att): #Need to lowercase when first read from the user
        mask = torch.zeros(logits.size()[2], dtype = torch.bool).to(self.device)
        try:
            i = self.att_list.index(att)
        except ValueError:
            logger.error(f'The given attribute {att} not supported in the given model {self.model_path}')
            raise
        mask[self.group_ids[i]] = True
        logits_g = logits[:,:,mask]
        pred_ids = torch.argmax(logits_g,dim=-1)
        pred_ids = pred_ids.cpu().apply_(lambda x: self.group_ids[i][x])
        pred = self.processor.batch_decode(pred_ids,spaces_between_special_tokens=True)[0].split()
        return list(map(lambda x:{f'p_{att}':'+',f'n_{att}':'-'}[x], pred))

    def read_audio_file(self, audio_file):
        if not os.path.exists(audio_file):
            logger.error(f'Audio file {audio_file} is not exist')
            raise FileNotFoundError
        y, _ = librosa.load(audio_file, sr=self.sampling_rate)

        return y


    def get_logits(self, y):
        
        input_values = self.processor(audio=y, sampling_rate=self.sampling_rate, return_tensors="pt").input_values.to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_values).logits

        return logits


    def check_identical_phonemes(self, df_p2att):        
        identical_phonemes = []
        for index,row in df_p2att.iterrows():
            mask = df_p2att.eq(row).all(axis=1)    
            indexes = df_p2att[mask].index.values
            if len(indexes) > 1:
                identical_phonemes.append(tuple(indexes))
        if identical_phonemes:
            logger.warning('The following phonemes has identical phonological features given the phonological features used in the model. If using fixed weight layer, these phonemes will be confused with each other')
            identical_phonemes = set(identical_phonemes)
            for x in identical_phonemes:
                logger.warning(f"{','.join(x)}")

    def read_phoneme2att(self,p2att_file):

        if not os.path.exists(p2att_file):
            logger.error(f'Phonological matrix file {p2att_file} is not exist')
            raise FileNotFoundError(f'{p2att_file}')
        
        df_p2att = pd.read_csv(p2att_file, index_col=0)
        
        self.check_identical_phonemes(df_p2att)
        not_supported = set(df_p2att.columns) - set(self.att_list)
        if not_supported:
            logger.warning(f"Attribute/s {','.join(not_supported)} is not supported by the model {self.model_path} and will be ignored. To get available attributes of the selected model run transcribe --model_path=/path/to/model print_availabel_attributes")
            df_p2att = df_p2att.drop(columns=not_supported)
        
        self.phoneme_list = df_p2att.index.values
        self.p2att_map = {}
        for i, r in df_p2att.iterrows():
            phoneme = i
            self.p2att_map[phoneme] = []
            for att in r.index.values:
                if f'p_{att}' not in self.processor.tokenizer.vocab:
                    logger.warn(f'Attribute {att} is not supported by the model {self.model_path} and will be ignored. To get available attributes of the selected model run transcribe --model_path=/path/to/model print_availabel_attributes')
                    continue
                value = r[att]
                if value == 0:
                    self.p2att_map[phoneme].append(f'n_{att}')
                elif value == 1:
                    self.p2att_map[phoneme].append(f'p_{att}')
                else:
                    logger.error(f'Invalid value of {value} for attribute {att} of phoneme {phoneme}. Values in the phoneme to attribute map should be either 0 or 1')
                    raise ValueError(f'{value} should be 0 or 1')


    def create_phoneme_tokenizer(self):
        vocab_list = self.phoneme_list
        vocab_dict = {v: k+1 for k, v in enumerate(vocab_list)}
        vocab_dict['<pad>'] = 0
        vocab_dict = dict(sorted(vocab_dict.items(), key= lambda x: x[1]))
        vocab_file = 'phoneme_vocab.json'
        with open(vocab_file, 'w') as f:
            json.dump(vocab_dict, f)
        #Build processor
        self.phoneme_tokenizer = Wav2Vec2CTCTokenizer(vocab_file, pad_token="<pad>", word_delimiter_token="")
        
    def create_phonological_matrix(self):
        self.phonological_matrix = torch.zeros((self.phoneme_tokenizer.vocab_size, self.processor.tokenizer.vocab_size)).type(torch.FloatTensor).to(self.device)
        self.phonological_matrix[self.phoneme_tokenizer.pad_token_id, self.processor.tokenizer.pad_token_id] = 1
        for p in self.phoneme_list:
            for att in self.p2att_map[p]:
                self.phonological_matrix[self.phoneme_tokenizer.convert_tokens_to_ids(p), self.processor.tokenizer.convert_tokens_to_ids(att)] = 1
            

    #This function gets the attribute logits from the output layer and convert to phonemes
    #Input is a sequence of logits (one vector per frame) and output phoneme sequence
    #Note that this is CTC so number of output phonemes is not equal to number of input frames
    def decode_phoneme(self,logits):
        def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
            if mask is not None:
                mask = mask.float()
                while mask.dim() < vector.dim():
                    mask = mask.unsqueeze(1)
                # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
                # results in nans when the whole vector is masked.  We need a very small value instead of a
                # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
                # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
                # becomes 0 - this is just the smallest value we can actually use.
                vector = vector + (mask + 1e-45).log()
            return torch.nn.functional.log_softmax(vector, dim=dim)
        
        log_props_all_masked = []
        for i in range(len(self.att_list)):
            mask = torch.zeros(logits.size()[2], dtype = torch.bool).to(self.device)
            mask[self.group_ids[i]] = True
            mask.unsqueeze_(0).unsqueeze_(0)
            log_probs = masked_log_softmax(vector=logits, mask=mask, dim=-1).masked_fill(~mask,0)
            log_props_all_masked.append(log_probs)
        log_probs_cat = torch.stack(log_props_all_masked, dim=0).sum(dim=0)
        log_probs_phoneme = torch.matmul(self.phonological_matrix,log_probs_cat.transpose(1,2)).transpose(1,2).type(torch.FloatTensor)
        pred_ids = torch.argmax(log_probs_phoneme,dim=-1)
        pred = self.phoneme_tokenizer.batch_decode(pred_ids,spaces_between_special_tokens=True)[0]
        return pred

    
    def print_human_readable(self, output, with_phoneme = False):
            column_widths = []
            rows = []
            if with_phoneme:
                column_widths.append(max([len(att['Name']) for att in output['Attributes']]+[len('Phoneme')]))
                column_widths.extend([5]*max([len(att['Pattern']) for att in output['Attributes']]+[len(output['Phoneme']['symbols'])]))
                rows.append(('Phoneme'.center(column_widths[0]), *[s.center(column_widths[j+1]) for j,s in enumerate(output['Phoneme']['symbols'])]))
            else:
                column_widths.append(max([len(att['Name']) for att in output['Attributes']]))
                column_widths.extend([5]*max([len(att['Pattern']) for att in output['Attributes']]))
            for i in range(len(output['Attributes'])):
                att = output['Attributes'][i]
                rows.append((att['Name'].center(column_widths[0]), *[s.center(column_widths[j+1]) for j,s in enumerate(att['Pattern'])]))
            out_string = ''
            for row in rows:
                out_string += '|'.join(row)
                out_string += '\n'
            return out_string

    #This function will do the followings:
    #1- if phonological_matrix_file=None and phoneme = False and attribute = None --> raise error nothing to be done
    #2- if phonological_matrix_file=None and phoneme = True and attribute = None --> raise error phonological_matrix_file is needed to recognize phoneme from attribute
    #3- if phonological_matrix_file=path/to/file and phoneme = True and attribute = None --> do batch phoneme recognition
    #4- if attribute = 'all' or 'list' do batch recognize of attributes
    #4-1 if phonological_matrix_file create also refrence attribute
    
    def transcribe_dataset(self, 
                           input_dataset_path,
                           output_dataset_path,
                           split=None,
                           phonological_matrix_file=None, 
                           recognize_phoneme=True, 
                           attributes=None):

        dataset = load_from_disk(input_dataset_path)
        if isinstance(dataset, DatasetDict):
            if split:
                if isinstance(split, str):
                    dataset = dataset[split]
                if isinstance(split, tuple):
                    dataset = DatasetDict(dict([(k,dataset[k]) for k in split]))
                    
        
        
        def decode_batch(batch):
            logits = self.get_logits(y=batch['audio']['array'])
            batch['pred_phoneme'] = self.decode_phoneme(logits)
            return batch

        
        if recognize_phoneme:
            if not phonological_matrix_file:
                logger.error("phonological matrix file is needed to map attributes to phonemes")
                raise FileNotFoundError("No phonological matrix file is given")
            
            self.read_phoneme2att(phonological_matrix_file)
            self.create_phoneme_tokenizer()
            self.create_phonological_matrix()

            dataset_pred = dataset.map(decode_batch)
        
        dataset_pred.save_to_disk(output_dataset_path)


    
    
    def transcribe(self, audio_file, 
                   attributes='all', 
                   phonological_matrix_file = None, 
                   human_readable = True):

        
        output = {}
        output['wav_file_path'] = audio_file
        output['Attributes'] = []
        output['Phoneme'] = {}
        
        #Initiate the model
        #self.load_model()
        #self.get_available_attributes()
        #self.get_att_binary_group_indexs()

        if attributes == 'all':
            target_attributes = self.att_list
        else:
            attributes = attributes if isinstance(attributes,tuple) else (attributes,)
            target_attributes = [att.lower() for att in attributes if att.lower() in self.att_list]
        
        if not target_attributes:
            logger.error(f'None of the given attributes is supported by model {self.model_path}. To get available attributes of the selected model run transcribe --model_path=/path/to/model get_available_attributes')
            raise ValueError("Invalid attributes")

        #Process audio
        y = self.read_audio_file(audio_file)
        self.logits = self.get_logits(y)
        
        for att in target_attributes:
            output['Attributes'].append({'Name':att, 'Pattern' : self.decode_att(self.logits, att)})

        if phonological_matrix_file:
            self.read_phoneme2att(phonological_matrix_file)
            self.create_phoneme_tokenizer()
            self.create_phonological_matrix()
            output['Phoneme']['symbols'] = self.decode_phoneme(self.logits).split()
            


        json_string = json.dumps(output, indent=4)
        if human_readable:
            return self.print_human_readable(output, phonological_matrix_file!=None)
        else:
            return json_string
        #return json_string


    def evaluate_dataset(self, input_dataset_path,
                         split=None,
                         pred_phoneme='pred_phoneme',
                         ref_phoneme='phoneme',
                         metric_path='metrics/wer.py',
                         confusion_matrix=True,
                         diph2mono_file=None,
                         decouple = True): #If diph2mono is provided and decouple = True, both ref and pred phonemes will be decoupled, if decouple = False
                                          #both of them will be coupled.
        
        diphthong_str = 'no_diph' if decouple else 'diph'
        def _load_diphthongs_to_monophthongs_map(diphthongs_to_monophthongs_map_file):
            with open(diphthongs_to_monophthongs_map_file, 'r') as f:
                self.diphthongs_to_monophthongs_map = dict([(x.split(',')[0], ' '.join(x.split(',')[1:])) for x in f.read().splitlines()])
                self.monophthongs_to_diphthongs_map = dict([(v,k) for k,v in self.diphthongs_to_monophthongs_map.items()])
        
        def _process_diphthongs(batch, phoneme_column, decouple=True):
            if decouple:
                mapper = self.diphthongs_to_monophthongs_map
            else:
                mapper = self.monophthongs_to_diphthongs_map
            
            pattern = r'|'.join([f'\\b{x}\\b' for x in mapper.keys()])
            batch[phoneme_column] = re.sub(pattern, lambda x: x.group(0).replace(x.group(0).strip(),mapper[x.group(0).strip()]), batch[phoneme_column])
            return batch
                    
        
        metric = evaluate.load(metric_path)
        dataset = load_from_disk(input_dataset_path)

        if diph2mono_file:
            _load_diphthongs_to_monophthongs_map(diph2mono_file)
            for phoneme_column in [pred_phoneme, ref_phoneme]:
                dataset = dataset.map(_process_diphthongs, fn_kwargs={'phoneme_column':phoneme_column,'decouple':decouple}, load_from_cache_file=False)
                    
                    
        if isinstance(dataset, DatasetDict):
            if split:
                if isinstance(split, str):
                    dataset = dataset[split]
                    column_names = dataset.column_names
                if isinstance(split, tuple):
                    dataset = DatasetDict(dict([(k,dataset[k]) for k in split]))
                    column_names = dataset[split[0]].column_names
            else:
                column_names = dataset[list(dataset.keys())[0]].column_names
        else:
            column_names = dataset.column_names
                    
        def evaluate_batch(batch):
            wer_val = metric.compute(predictions=batch[pred_phoneme], references= batch[ref_phoneme])
            return {'wer':[wer_val]}

        dataset_eval = dataset.map(evaluate_batch, batched=True, batch_size= None, remove_columns=column_names, load_from_cache_file=False)
        #Compute confusion matrix
        if confusion_matrix:
            cm_output_basename = os.path.normpath(input_dataset_path)
            cm_metric = cm.phoneme_confusion_matrix()
            if isinstance(dataset, DatasetDict):
                for split in dataset.keys():
                    cm_output_file = '_'.join([cm_output_basename, split]) + f'{diphthong_str}_cm.csv'
                    _ = cm_metric.compute(dataset[split][ref_phoneme], dataset[split][pred_phoneme])
                    cm_metric.save_cm(cm_output_file)
            else:
                cm_output_file = f'{cm_output_basename}_{diphthong_str}_cm.csv'
                _ = cm_metric.compute(dataset[ref_phoneme], dataset[pred_phoneme])
                cm_metric.save_cm(cm_output_file)
        
        if isinstance(dataset_eval, DatasetDict):
            output = {}
            for split in dataset_eval.keys():
                output[split] = dataset_eval[split][0]["wer"]
            return output
        else:
            return dataset_eval[0]["wer"]
            

def main():
    fire.Fire(transcribe_SA)

if __name__ == '__main__':
    main()
