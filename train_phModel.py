import fire
import logging
import sys
import yaml
import pandas as pd
from itertools import chain
from os import makedirs
from os.path import join
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, HubertForCTC, WavLMForCTC, Trainer, TrainingArguments
from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset, load_metric, ClassLabel, load_from_disk, DatasetDict, concatenate_datasets, Dataset
import evaluate
import re

logger = logging.getLogger(__name__)
# Setup logging
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formater = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",)
console_handler.setFormatter(formater)
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)

SUPPORTED_MODELS = ['WavLM','WAV2VEC2','HuBERT']


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """


    processor: Wav2Vec2Processor
    padding_features: Union[bool, str] = True
    padding_labels: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding_features,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_batch = self.processor.pad(
                labels=label_features,
                padding=self.padding_labels,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )
        
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


class TrainPhModel():
    def __init__(self, config_file):
        # Read YAML file
        logger.info('Init Object')
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.error(exc)
                raise
        if torch.cuda.is_available():
            self.accelerate = True
            self.device = torch.device('cuda')
            self.n_devices = torch.cuda.device_count()
            assert self.n_devices == 1, 'Support only single GPU. Please use CUDA_VISIBLE_DEVICES=gpu_index if you have multiple gpus' #Currently support only single gpu
        else:
            self.device = torch.device('cpu')
            self.n_devices = 1
        try:    
            self.working_dir = config['output']['working_dir']
            makedirs(self.working_dir, exist_ok=True)
            file_handler = logging.FileHandler(join(self.working_dir,'log'))
            file_handler.setFormatter(formater)
            file_handler.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.error(f'Failed to create working dir in {self.working_dir}')
            raise
        
        logger.info('Loading Configuration...')
        self.dataset_path = config['datasets']['data_path']
        self.train_part = [x.strip() for x in config['datasets']['train_part'].split(',')]
        self.validation_part = [x.strip() for x in config['datasets']['validation_part'].split(',')]
        self.test_part = [x.strip() for x in config['datasets']['test_part'].split(',')]
        self.cache_dir = config['datasets']['cache_dir']
        self.phoneme_list_file = config['datasets']['phoneme_list_file']

        #self.attribute_list_file = config['phonological']['attribute_list_file']
        #self.phoneme2att_map_file = config['phonological']['phoneme2att_map_file']
        #self.phonetic_alphabet = config['phonological']['phonetic_alphabet']

        self.sampling_rate = config['preprocessor']['sampling_rate']
        self.do_normalize = config['preprocessor']['do_normalize']
        self.return_attention_mask = config['preprocessor']['return_attention_mask']
        self.do_phonemize = config['preprocessor']['do_phonemize']
        self.phoneme_column = config['preprocessor']['phoneme_column']
        self.num_proc = config['preprocessor']['num_proc']
        self.save_preprocessed_data = config['preprocessor']['save_preprocessed_data']
        self.load_from_preprocessed_data = config['preprocessor']['load_from_preprocessed_data']
        self.max_length_in_sec = config['preprocessor']['max_length_in_sec']
        self.decouple_diphthongs = config['preprocessor']['decouple_diphthongs']
        self.diphthongs_to_monophthongs_map_file = config['preprocessor'].get('diphthongs_to_monophthongs_map_file','')

        self.model_path = config['training']['model_path']
        self.model_type = config['training'].get('model_type',"WAV2VEC2")
        self.gradient_checkpointing = config['training']['gradient_checkpointing']
        self.ctc_loss_reduction = config['training']['ctc_loss_reduction']
        self.freeze_feature_encoder = config['training']['freeze_feature_encoder']
        self.group_by_length = config['training']['group_by_length']
        self.train_batch_size = config['training']['train_batch_size']
        self.evaluation_strategy = config['training']['evaluation_strategy']
        self.enable_fp16 = config['training']['enable_fp16']
        self.num_train_epochs = config['training']['num_train_epochs']
        self.save_steps = config['training']['save_steps']
        self.logging_steps = config['training']['logging_steps']
        self.prediction_loss_only = config['training']['prediction_loss_only']
        self.learning_rate = float(config['training']['learning_rate'])
        self.weight_decay = config['training']['weight_decay']
        self.warmup_ratio = config['training']['warmup_ratio']
        self.load_best_model_at_end = config['training']['load_best_model_at_end']
        self.save_total_limit = config['training']['save_total_limit']
        
        self.trained_model_path = config['evaluation'].get('trained_model_path', join(self.working_dir,'fine_tune','best'))
        self.spaces_between_special_tokens = config['evaluation']['spaces_between_special_tokens']
        self.metric_path = config['evaluation']['metric_path']
        self.eval_extra_data = config['evaluation'].get('eval_extra_data', '')
        self.eval_extra_data_parts = config['evaluation'].get('eval_extra_data_parts', '').split(',')
        self.auto_eval = config['evaluation']['auto_eval']
        self.eval_extra_data_phoneme_col = config['evaluation'].get('eval_extra_data_phoneme_col', self.phoneme_column)

        #Load Phoneme List File
        try:
            with open(self.phoneme_list_file,'r') as f:
                self.phoneme_list = f.read().splitlines()
        except FileNotFoundError:
            logger.error(f'Phoneme list file {self.phoneme_list_file} not exist')
            raise
        self.metric = evaluate.load(self.metric_path)

        
    def create_processor(self):
        vocab_list = self.phoneme_list
        vocab_dict = {v: k+1 for k, v in enumerate(vocab_list)}
        vocab_dict['<pad>'] = 0
        vocab_dict = dict(sorted(vocab_dict.items(), key= lambda x: x[1]))
        self.vocab_file = join(self.working_dir,'vocab.json')
        with open(self.vocab_file, 'w') as f:
            json.dump(vocab_dict, f)
        #Build processor
        self.tokenizer = Wav2Vec2CTCTokenizer(self.vocab_file, pad_token="<pad>", word_delimiter_token="")
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=self.sampling_rate, padding_value=0.0, do_normalize=self.do_normalize, return_attention_mask=self.return_attention_mask)
        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)


    #Should run with batched=True
    def _prepare_dataset(self, batch):
        # check that all files have the same sampling rate
        sampling_rates = set([i['sampling_rate'] for i in batch["audio"]])
        assert (
            len(sampling_rates) == 1 and list(sampling_rates)[0] == 16000
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    
        batch["input_values"] = self.processor(audio=[i['array'] for i in batch['audio']], sampling_rate=batch['audio'][0]["sampling_rate"]).input_values

        batch["labels"] = self.processor(text=batch[self.phoneme_column]).input_ids
        return batch

    def _load_diphthongs_to_monophthongs_map(self):
        with open(self.diphthongs_to_monophthongs_map_file, 'r') as f:
            self.diphthongs_to_monophthongs_map = dict([(x.split(',')[0], ' '.join(x.split(',')[1:])) for x in f.read().splitlines()])
            self.monophthongs_to_diphthongs_map = dict([(v,k) for k,v in self.diphthongs_to_monophthongs_map.items()])
    

    def _process_diphthongs(self,batch, phoneme_column, decouple=True):
        if decouple:
            mapper = self.diphthongs_to_monophthongs_map
        else:
            mapper = self.monophthongs_to_diphthongs_map                                                     
        pattern = r'|'.join([f'\\b{x}\\b' for x in mapper.keys()])
        batch[phoneme_column] = re.sub(pattern, lambda x: x.group(0).replace(x.group(0).strip(),mapper[x.group(0).strip()]), batch[phoneme_column])
        return batch


    def load_data(self):
        #TODO if data is dataset and not dictdataset use automatic train,test,valid split
        
        #TODO check if dict and train part not specified raise error
        data_loaded = train_data_loaded = valid_data_loaded = test_data_loaded = False
        if self.load_from_preprocessed_data:
            try:
                self.data_train = load_from_disk(join(self.working_dir,'preprocessed_data','train'))
                logger.info('Training data loaded from preprocessed version')
                train_data_loaded = True
            except:
                logger.warning('Failed to load Training data from preprocessed version. Trying to reprocess from original source')

        if not train_data_loaded:
            if not data_loaded:
                try:
                    data = load_from_disk(self.dataset_path)
                    data_loaded = True
                except Exception as e:
                    logger.error(f'Failed to load data at {self.dataset_path}')
                    raise

            tmp_list = []
            try:
                for k in self.train_part:
                    tmp_list.append(data[k])
                data_train = concatenate_datasets(tmp_list)
                self.data_train = self.preprocess_data(data_train, bTraining=True)
                if self.save_preprocessed_data:
                     self.data_train.save_to_disk(join(self.working_dir,'preprocessed_data','train'))
            except KeyError:
                logger.error('One or more of the train parts specified in yaml file are not exist in the dataset')
                raise

        if self.load_from_preprocessed_data:
            try:
                self.data_valid = load_from_disk(join(self.working_dir,'preprocessed_data','valid'))
                logger.info('Validation data loaded from preprocessed version')
                valid_data_loaded = True
            except:
                logger.warning('Failed to load Validation data from preprocessed version. Trying to reprocess from original source')

        if not valid_data_loaded:
            if not data_loaded:
                try:
                    data = load_from_disk(self.dataset_path)
                    data_loaded = True
                except Exception as e:
                    logger.error(f'Failed to load data at {self.dataset_path}')
                    raise
            tmp_list = []
            try:
                for k in self.validation_part:
                    tmp_list.append(data[k])
                data_valid = concatenate_datasets(tmp_list)
                self.data_valid = self.preprocess_data(data_valid, bTraining=True)
                if self.save_preprocessed_data:
                     self.data_valid.save_to_disk(join(self.working_dir,'preprocessed_data','valid'))
            except KeyError:
                logger.error('One or more of the validation parts specified in yaml file are not exist in the dataset')
                raise


        if self.load_from_preprocessed_data:
            try:
                self.data_test = load_from_disk(join(self.working_dir,'preprocessed_data','test'))
                logger.info('Test data loaded from preprocessed version')
                test_data_loaded = True
            except:
                logger.warning('Failed to load Test data from preprocessed version. Trying to reprocess from original source')

        if not test_data_loaded:        
            if not data_loaded:
                try:
                    data = load_from_disk(self.dataset_path)
                    data_loaded = True
                except Exception as e:
                    logger.error(f'Failed to load data at {self.dataset_path}')
                    raise

            try:
                data_test = DatasetDict(dict([(k,data[k]) for k in self.test_part]))
                self.data_test = self.preprocess_data(data_test, bTraining=False)
                if self.save_preprocessed_data:
                     self.data_test.save_to_disk(join(self.working_dir,'preprocessed_data','test'))
            except KeyError:
                logger.warning('One or more of the test parts specified in yaml file are not exist in the dataset, test data will be ignored')
        
    
    def preprocess_data(self,
                        data,
                        bTraining=True):
        
        data = data.filter(lambda x:len(x['audio']['array']) < self.max_length_in_sec*x['audio']['sampling_rate'], num_proc=self.num_proc)
        
        if self.decouple_diphthongs:
            if self.diphthongs_to_monophthongs_map_file:
                self._load_diphthongs_to_monophthongs_map()
            else:
                logger.error("decouple_diphthongs is set to True but to mapping file is provided, please explicitly set decouple_diphthongs to False or provide a mapping file in diphthongs_to_monophthongs_map_file")
                raise FileNotFoundError
            data = data.map(self._process_diphthongs, batched=False, fn_kwargs={'phoneme_column':self.phoneme_column, 'decouple':True}, load_from_cache_file=False)
        
        if bTraining:
            data = data.map(self._prepare_dataset, remove_columns=data.column_names, batch_size=8, num_proc=self.num_proc, batched=True)
        
        return data

    def prepare_trainer(self):

        def _compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)
        
            pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        
            pred_str = self.processor.batch_decode(pred_ids, spaces_between_special_tokens=self.spaces_between_special_tokens)
            label_str = self.processor.batch_decode(pred.label_ids, spaces_between_special_tokens=self.spaces_between_special_tokens)
        
            per = self.metric.compute(predictions=pred_str, references=label_str)
            return {"wer": per}

        
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding_labels=True, max_length_labels=None)

        if self.model_type=='WAV2VEC2':
            self.model = Wav2Vec2ForCTC.from_pretrained(
                self.model_path,
                gradient_checkpointing=self.gradient_checkpointing,
                ctc_loss_reduction=self.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
                cache_dir=self.cache_dir
                ).to(self.device)
        elif self.model_type=='WavLM':
            self.model = WavLMForCTC.from_pretrained(
                self.model_path,
                gradient_checkpointing=self.gradient_checkpointing,
                ctc_loss_reduction=self.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
                cache_dir=self.cache_dir
                ).to(self.device)
        elif self.model_type == 'HuBERT':
            self.model = HubertForCTC.from_pretrained(
                self.model_path,
                gradient_checkpointing=self.gradient_checkpointing,
                ctc_loss_reduction=self.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
                cache_dir=self.cache_dir
                ).to(self.device)
        else:
            logger.error(f'model type {self.model_type} not supported. Should be on of {" ".join(SUPPORTED_MODELS)}')
            raise ValueError(f'Unsupported model type {self.model_type}')
        
        if self.freeze_feature_encoder:
            self.model.freeze_feature_encoder()

        self.training_args = TrainingArguments(
          output_dir=join(self.working_dir,'fine_tune'),
          group_by_length=self.group_by_length,
          per_device_train_batch_size=int(self.train_batch_size/self.n_devices),
          evaluation_strategy=self.evaluation_strategy,
          fp16=self.enable_fp16,
          num_train_epochs=self.num_train_epochs,
          save_steps=self.save_steps,
          logging_steps=self.logging_steps,
          prediction_loss_only=self.prediction_loss_only,
          learning_rate=self.learning_rate,
          weight_decay=self.weight_decay,
          warmup_ratio=self.warmup_ratio,
          load_best_model_at_end=self.load_best_model_at_end,
          save_total_limit=self.save_total_limit,
        )

        self.trainer = Trainer(
            model=self.model,
            data_collator=self.data_collator,
            args=self.training_args,
            compute_metrics=_compute_metrics, 
            train_dataset=self.data_train,
            eval_dataset=self.data_valid,
            tokenizer=self.processor.feature_extractor,
            )

    def save_model(self):
        self.saved_model_path = join(self.working_dir, 'fine_tune','best')
        self.model.save_pretrained(self.saved_model_path)
        self.processor.save_pretrained(self.saved_model_path)


    def train_model(self, resume_from_checkpoint=False):
        self.create_processor()
        self.load_data()
        self.prepare_trainer()
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.save_model()
        if self.auto_eval:
            self.evaluate_model()

    def map_to_result(self, batch):
        input_values = self.processor(
              batch['audio']['array'],
              sampling_rate=batch['audio']["sampling_rate"],
              return_tensors="pt").input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
    
        pred_ids = torch.argmax(logits, dim=-1)
        batch["pred_str"] = self.processor.batch_decode(pred_ids,spaces_between_special_tokens=True)[0]
    
        return batch
        
    def evaluate_model(self,
                          eval_data=None,
                          eval_parts=None,
                          suffix=None,
                          phoneme_column=None,
                          decouple_diph=None):
        
        #Load the model and processor deafult at working_dir/fine_tune/best could be overridden from the yaml file by setting 
        #value for evaluation->trained_model_path
        logger.info(f'Load Model for evaluation from {self.trained_model_path}')
        self.processor = Wav2Vec2Processor.from_pretrained(self.trained_model_path)
        if self.model_type == "WAV2VEC2":
            self.model = Wav2Vec2ForCTC.from_pretrained(self.trained_model_path)
        elif self.model_type == "WavLM":
            self.model = WavLMForCTC.from_pretrained(self.trained_model_path)
        elif self.model_type == "HuBERT":
            self.model = HubertForCTC.from_pretrained(self.trained_model_path)
        else:
            logger.error(f'model type {self.model_type} not supported. Should be on of {" ".join(SUPPORTED_MODELS)}')
            raise ValueError(f'Unsupported model type {self.model_type}')

        self.model.eval()
        #self.model.to(self.device)
        
        test_data_loaded = False
        
        if eval_data:
            if phoneme_column:
                self.phoneme_column=phoneme_column
            try:
                data = load_from_disk(eval_data)
            except Exception as e:
                logger.error(f'Failed to load data at {eval_data}')
                raise
            if eval_parts:
                logger.info(f'Performing evaluation of {eval_parts} of {eval_data}')
                try:
                    eval_parts = eval_parts if isinstance(eval_parts,tuple) else (eval_parts,)
                    data_test = DatasetDict(dict([(k,data[k]) for k in eval_parts]))
                except KeyError:
                    logger.warning('One or more of the given eval parts  are not exist in the dataset, missing data will be ignored')
                self.data_test = self.preprocess_data(data_test, bTraining=False)
                test_data_loaded = True
            else:
                logger.info(f'Performing evaluation of {eval_data}')
                self.data_test = self.preprocess_data(data, bTraining=False)
                test_data_loaded = True
        else: #If eval_data not passed from the command line, check if the dataset is already loaded. 
            if hasattr(self,'data_test'): #The object already has test data loaded, this case when load_data already called for training
                if isinstance(self.data_test, DatasetDict) or isinstance(self.data_test, Dataset):
                    logger.info(f'Performing evaluation of {self.test_part} of {self.dataset_path}')
                    test_data_loaded = True

        if not test_data_loaded:
            if self.eval_extra_data:
                self.phoneme_column = self.eval_extra_data_phoneme_col
                try:
                    data = load_from_disk(self.eval_extra_data)
                except Exception as e:
                    logger.error(f'Failed to load data at {self.eval_extra_data}')
                    raise
                if self.eval_extra_data_parts[0]:
                    logger.info(f'Performing evaluation of {self.eval_extra_data_parts} of {self.eval_extra_data}')
                    try:
                        data_test = DatasetDict(dict([(k,data[k]) for k in self.eval_extra_data_parts]))
                    except KeyError:
                        logger.warning('One or more of the eval parts specified in yaml file are not exist in the dataset, missing data will be ignored')
                    self.data_test = self.preprocess_data(data_test, bTraining=False)
                    test_data_loaded = True
                else:
                    logger.info(f'Performing evaluation of {self.eval_extra_data}')
                    self.data_test = self.preprocess_data(data, bTraining=False)
                    test_data_loaded = True
        
        if test_data_loaded:
            logger.info(f'Phonemes for evaluation read from {self.phoneme_column} column')
            self.model.to(self.device)
            isdict = isinstance(self.data_test, DatasetDict)

            if suffix:
                suffix = f"{suffix}_{'_'.join(self.data_test.keys()) if isdict else 'testset'}"
            else:
                suffix = '_'.join(self.data_test.keys()) if isdict else 'testset'
            
            self.results = self.data_test.map(self.map_to_result, batched=False, load_from_cache_file=False)
            if decouple_diph!= None:
                self._load_diphthongs_to_monophthongs_map()
                for col in ["pred_str", self.phoneme_column]:
                    self.results = self.results.map(self._process_diphthongs, fn_kwargs={'phoneme_column':col,'decouple':decouple_diph}, load_from_cache_file=False)

            self.results.save_to_disk(join(self.working_dir,f"results_{suffix}.db"))
    
            #metric = evaluate.load(self.metric_path)
            
            
            with open(join(self.working_dir,f"results_{suffix}.txt"),'w') as f:
                if isdict:
                    for dataset in self.results:
                        print("Test PER: {:.3f}".format(self.metric.compute(predictions=self.results[dataset]["pred_str"], references=self.results[dataset][self.phoneme_column])),file=f)
                else:
                    print("Test PER: {:.3f}".format(self.metric.compute(predictions=self.results["pred_str"], references=self.results[self.phoneme_column])),file=f)
            logger.info(f'Results dataset saved in {join(self.working_dir,f"results_{suffix}.db")} and the results saved in {join(self.working_dir,f"results_{suffix}.txt")}')
        


def main():
    fire.Fire(TrainPhModel)

if __name__ == '__main__':
    main()
