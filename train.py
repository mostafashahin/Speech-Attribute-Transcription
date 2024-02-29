import fire
import logging
import sys
import yaml
import pandas as pd
from itertools import chain
from os import makedirs
from os.path import join
import json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Trainer, TrainingArguments
from torch import nn
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from datasets import load_dataset, load_metric, ClassLabel, load_from_disk, DatasetDict, concatenate_datasets, Dataset
import evaluate


logger = logging.getLogger(__name__)
# Setup logging
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formater = logging.Formatter(fmt="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",)
console_handler.setFormatter(formater)
console_handler.setLevel(logging.INFO)

logger.addHandler(console_handler)


@dataclass
class DataCollatorMCTCWithPadding:
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
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding_features,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        labels_list = []
        nGroups = len(features[0]["labels"])
        for i in range(nGroups):
            label_features = [{"input_ids": feature["labels"][i]} for feature in features]
            labels_batch = self.processor.tokenizer.pad(
                        label_features,
                        padding=self.padding_labels,
                        max_length=self.max_length_labels,
                        pad_to_multiple_of=self.pad_to_multiple_of_labels,
                        return_tensors="pt",
                    )
            labels_tmp = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100).unsqueeze(dim=1)
            labels_list.append(labels_tmp)

        batch["labels"] = torch.cat(labels_list,dim=1)

        return batch



class SCTCTrainer(Trainer):
    def __init__(self, **kargs):
        self.group_ids = kargs.pop('group_ids') #List with number of items in each group
        super(SCTCTrainer, self).__init__(**kargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(inputs.get('input_values'))
        logits = outputs.get('logits')
        ngroups = len(self.group_ids)
        #first two tokens (0,1) for <pad> and <unk>
        assert labels.dim() == 3, "in multi-label 3D tensor is expected"
        assert ngroups == labels.size()[1], "Second dim should match number of groups"

        #IMPORTANT 0 reserved to <pad>  shared among all groups #VALIDATE THIS?!
        #IMPORTANT STARTING FROM 1, 1:1+n IS THE n ELEMENTS in FIRST GROUP and from 1+n:1+n+m IS THE M ELEMENTS IN SECOND GROUP
        #start_indx = 1 #0  for <pad>
        all_losses = []
        for i in range(ngroups):
            mask = torch.zeros(logits.size()[2], dtype = torch.bool)
            mask[0] = True
            mask[list(self.group_ids[i].keys())] = True


            targets = labels[:,i,:].squeeze()
            g_logits = logits[:,:,mask]
            log_probs = nn.functional.log_softmax(g_logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            #Label padding = -100
            labels_mask = targets >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = targets.masked_select(labels_mask)
            flattened_targets = flattened_targets.cpu().apply_(lambda x: self.group_ids[i][x]) #So all targets will start from index 1
            flattened_targets = flattened_targets.to(self.args.device)
            input_lengths = model._get_feat_extract_output_lengths(torch.ones_like(inputs.get('input_values'),dtype=torch.int32).sum(-1))
            loss = F.ctc_loss(log_probs, flattened_targets, input_lengths, target_lengths, blank=model.config.pad_token_id, zero_infinity=model.config.ctc_zero_infinity, reduction=model.config.ctc_loss_reduction)
            all_losses.append(loss)
        sctc_loss = sum(all_losses) #TODO: consider average over number of groups NOT VALID
        #TODO: consider reduction over input_lengths*target_lengths
        return (sctc_loss, outputs) if return_outputs else sctc_loss


class TrainSAModel():
    def __init__(self, config_file):
        # Read YAML file
        logger.info('Init Object')
        with open(config_file, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
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

        self.attribute_list_file = config['phonological']['attribute_list_file']
        self.phoneme2att_map_file = config['phonological']['phoneme2att_map_file']
        self.phonetic_alphabet = config['phonological']['phonetic_alphabet']

        self.sampling_rate = config['preprocessor']['sampling_rate']
        self.do_normalize = config['preprocessor']['do_normalize']
        self.return_attention_mask = config['preprocessor']['return_attention_mask']
        self.do_phonemize = config['preprocessor']['do_phonemize']
        self.phoneme_column = config['preprocessor']['phoneme_column']
        self.num_proc = config['preprocessor']['num_proc']
        self.save_preprocessed_data = config['preprocessor']['save_preprocessed_data']
        self.load_from_preprocessed_data = config['preprocessor']['load_from_preprocessed_data']
        self.max_length_in_sec = config['preprocessor']['max_length_in_sec']

        self.model_path = config['training']['model_path']
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

        
    def load_attribute_list(self):
        try:
            with open(self.attribute_list_file) as f:
                self.list_att = f.read().splitlines()
        except FileNotFoundError:
            logger.error(f'{self.attribute_list_file} is not found')
            raise FileNotFoundError(
                f"The list of attribute file {self.attribute_list_file} is not found" 
            )
    def load_p2a_map(self):
        try:
            with open(self.phoneme2att_map_file) as f:
                self.df_p2a = pd.read_csv(self.phoneme2att_map_file)
        except FileNotFoundError:
            logger.error(f'{self.phoneme2att_map_file} is not found')
            raise FileNotFoundError(
                f"The phoneme to attribute map file {self.phoneme2att_map_file} is not found" 
            )
        #Check all attributes are exists in the csv file
        if not set(self.list_att).issubset(set(self.df_p2a.columns)):
            logger.warning('Missing attributes in the phoneme2att map file')
            miss_attribute = set(self.list_att) - set(self.df_p2a.columns)
            logger.warning(f"The following attributes will be removed from the training {','.join(miss_attribute)}")

    #For each attribute create two symbols, p_att, n_att added in one group
    def create_binary_groups(self):
        self.groups = []
        for att in self.list_att:
            binary_att = [f'p_{att}',f'n_{att}'] #Each attribute could be +ve or -ve 
            self.groups.append(binary_att)

    #Map each phoneme to either n_att or p_att
    def create_phoneme_binary_mappers(self):
        self.phoneme_binary_mappers = []
        for g in self.groups:
            p2att = {}
            att = g[0].split('_')[1] #First one is 'p_att'
            p_att_phs = self.df_p2a[self.df_p2a[att]==1].index
            n_att_phs = self.df_p2a[self.df_p2a[att]==0].index
            for idx in p_att_phs:
                ph = self.df_p2a.iloc[idx][f'Phoneme_{self.phonetic_alphabet}']
                p2att[ph] = f'p_{att}'
            for idx in n_att_phs:
                ph = self.df_p2a.iloc[idx][f'Phoneme_{self.phonetic_alphabet}']
                p2att[ph] = f'n_{att}'
            self.phoneme_binary_mappers.append(p2att)

    #As we train a single model for all the attributes then at the inference time we separate them into groups
    #each with just p_att, n_att then there are two indexes for each symbol, global index used in training
    #and local index used in the inference time. This function maps each symbol either to the global index if bTraining=True
    #or local index if bTraining = False
    def get_att_group_indx_map(self,bTraining=True):
        #Get group ids dictionary
        group_ids = [sorted(self.processor.tokenizer.convert_tokens_to_ids(group)) for group in self.groups]
        if bTraining:
            group_ids = [dict([(x[1],x[0]+1) for x in list(enumerate(g))]) for g in group_ids]
        else:
            group_ids = [dict([(x[0]+1,x[1]) for x in list(enumerate(g))]) for g in group_ids]
        return group_ids
        
    def create_processor(self):
        vocab_list = list(chain(*self.groups))
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


    def _create_att_targets(self, batch): #Could be dp or sb
        def mapToken(phList, mappers=self.phoneme_binary_mappers):
            g_labels = []
            for mapper in mappers:
                g_label = []
                for p in phList.split():
                    assert p in mapper, "{0} not in mapper".format(p)
                    g_label.append(mapper[p])
                g_labels.append(' '.join(g_label))
            return g_labels
        batch["target_text"] = list(map(mapToken, batch[self.phoneme_column]))
        return batch


    #Should run with batched=True
    def _prepare_dataset(self, batch):
        # check that all files have the same sampling rate
        sampling_rates = set([i['sampling_rate'] for i in batch["audio"]])
        assert (
            len(sampling_rates) == 1 and list(sampling_rates)[0] == 16000
        ), f"Make sure all inputs have the same sampling rate of {processor.feature_extractor.sampling_rate}."
    
        def processPerGroup(item):
           #I did this because using just tokenizer(item) interpret "semivowel" to two tokens, semivowel and vowel
           #TODO use unique name and not part of each other
           labels = self.processor.tokenizer([t.split() for t in item], is_split_into_words=True).input_ids
           return labels
    
        batch["input_values"] = self.processor(audio=[i['array'] for i in batch['audio']], sampling_rate=batch['audio'][0]["sampling_rate"]).input_values

        batch["labels"] = list(map(processPerGroup, batch["target_text"]))
        return batch

    def load_data(self):
        try:
            data = load_from_disk(self.dataset_path)
        except Exception as e:
            logger.error(f'Failed to load data at {self.dataset_path}')
            raise
        #TODO if data is dataset and not dictdataset use automatic train,test,valid split
        
        #TODO check if dict and train part not specified raise error
        train_data_loaded = valid_data_loaded = test_data_loaded = False
        if self.load_from_preprocessed_data:
            try:
                self.data_train = load_from_disk(join(self.working_dir,'preprocessed_data','train'))
                logger.info('Training data loaded from preprocessed version')
                train_data_loaded = True
            except:
                logger.warning('Failed to load Training data from preprocessed version. Trying to reprocess from original source')

        if not train_data_loaded:
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
            try:
                data_test = DatasetDict(dict([(k,data[k]) for k in self.test_part]))
                self.data_test = self.preprocess_data(data_test, bTraining=False)
                if self.save_preprocessed_data:
                     self.data_test.save_to_disk(join(self.working_dir,'preprocessed_data','test'))
            except KeyError:
                logger.warning('One or more of the test parts specified in yaml file are not exist in the dataset, test data will be ignored')
        
    
    def preprocess_data(self, data, bTraining=True):
        
        data = data.filter(lambda x:len(x['audio']['array']) < self.max_length_in_sec*x['audio']['sampling_rate'], num_proc=self.num_proc)
        data = data.map(self._create_att_targets, batched=True, batch_size=8, num_proc=self.num_proc)
        if bTraining:
            data = data.map(self._prepare_dataset, remove_columns=data.column_names, batch_size=8, num_proc=self.num_proc, batched=True)
        return data

    def prepare_trainer(self):
        self.data_collator = DataCollatorMCTCWithPadding(processor=self.processor, padding_labels=True, max_length_labels=None)
        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_path,
            gradient_checkpointing=self.gradient_checkpointing,
            ctc_loss_reduction=self.ctc_loss_reduction,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=self.processor.tokenizer.vocab_size,
            cache_dir=self.cache_dir
            ).to(self.device)
        
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

        self.trainer = SCTCTrainer(
            model=self.model,
            group_ids=self.get_att_group_indx_map(bTraining=True),
            data_collator=self.data_collator,
            args=self.training_args,
            #compute_metrics=compute_metrics, #Need to be added
            train_dataset=self.data_train,
            eval_dataset=self.data_valid,
            tokenizer=self.processor.feature_extractor,
            )

    def save_model(self):
        self.saved_model_path = join(self.working_dir, 'fine_tune','best')
        self.model.save_pretrained(self.saved_model_path)
        self.processor.save_pretrained(self.saved_model_path)


    def train_SA_model(self, resume_from_checkpoint=False):
        self.load_attribute_list()
        self.load_p2a_map()
        self.create_binary_groups()
        self.create_phoneme_binary_mappers()
        self.create_processor()
        self.load_data()
        self.prepare_trainer()
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        self.save_model()
        if self.auto_eval:
            self.evaluate_SA_model()

    def map_to_result(self, batch):
        input_values = self.processor(
              batch['audio']['array'],
              sampling_rate=batch['audio']["sampling_rate"],
              return_tensors="pt").input_values.to(self.device)

        with torch.no_grad():
            logits = self.model(input_values).logits
    
        start_indx = 1
        pred = []
        group_ids = self.get_att_group_indx_map(bTraining=False)
        for i in range(len(group_ids)):
            mask = torch.zeros(logits.size()[2], dtype = torch.bool)
            mask[0] = True
            mask[list(group_ids[i].values())] = True
            logits_g = logits[:,:,mask]
            pred_ids = torch.argmax(logits_g,dim=-1)
            #pred_ids[pred_ids>0] += start_indx - 1
            #start_indx += utils.number_items_per_group[i]
            pred_ids = pred_ids.cpu().apply_(lambda x: group_ids[i].get(x,x))
            pred.append(self.processor.batch_decode(pred_ids,spaces_between_special_tokens=self.spaces_between_special_tokens)[0])

        batch["pred_str"] = pred
    
        return batch
        
    def evaluate_SA_model(self, eval_data=None, eval_parts=None):
        #Load the model and processor deafult at working_dir/fine_tune/best could be overridden from the yaml file by setting 
        #value for evaluation->trained_model_path
        self.processor = Wav2Vec2Processor.from_pretrained(self.trained_model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.trained_model_path)

        self.model.eval()
        #self.model.to(self.device)
        
        self.load_attribute_list()
        self.load_p2a_map()
        self.create_binary_groups()
        self.create_phoneme_binary_mappers()
        
        test_data_loaded = False
        if eval_data:
            try:
                data = load_from_disk(eval_data)
            except Exception as e:
                logger.error(f'Failed to load data at {eval_data}')
                raise
            if eval_parts:
                try:
                    data_test = DatasetDict(dict([(k,data[k]) for k in eval_parts.split(',')]))
                    self.data_test = self.preprocess_data(data_test, bTraining=False)
                    test_data_loaded = True
                except KeyError:
                    logger.warning('One or more of the eval parts specified in yaml file are not exist in the dataset, test data will be ignored')
            else:
                self.data_test = self.preprocess_data(data, bTraining=False)
                test_data_loaded = True

        
        if hasattr(self,'data_test'): #The object already has test data loaded, this case when load_data already called for training
            if isinstance(self.data_test, DatasetDict) or isinstance(self.data_test, Dataset):
                test_data_loaded = True

        if not test_data_loaded:
            if self.eval_extra_data:
                try:
                    data = load_from_disk(self.eval_extra_data)
                except Exception as e:
                    logger.error(f'Failed to load data at {self.eval_extra_data}')
                    raise
                if self.eval_extra_data_parts[0]:
                    try:
                        data_test = DatasetDict(dict([(k,data[k]) for k in self.eval_extra_data_parts]))
                        self.data_test = self.preprocess_data(data_test, bTraining=False)
                        test_data_loaded = True
                    except KeyError:
                        logger.warning('One or more of the eval parts specified in yaml file are not exist in the dataset, test data will be ignored')
                else:
                    self.data_test = self.preprocess_data(data, bTraining=False)
                    test_data_loaded = True
        
        if test_data_loaded:
            self.model.to(self.device)
            isdict = isinstance(self.data_test, DatasetDict)

            suffix = '_'.join(self.data_test.keys()) if isdict else 'testset'
            
            self.results = self.data_test.map(self.map_to_result, batched=False, load_from_cache_file=False)
            self.results.save_to_disk(join(self.working_dir,f"results_{suffix}.db"))
    
            metric = evaluate.load(self.metric_path)
            
            ngroups = len(self.groups)
            with open(join(self.working_dir,f"results_{suffix}.txt"),'w') as f:
                if isdict:
                    for dataset in self.results:
                        for g in range(ngroups):
                            pred = [item[g] for item in self.results[dataset]['pred_str']]
                            target = [item[g] for item in self.results[dataset]['target_text']]
                            print("{} group {} AER: {:.5f}".format(dataset,self.groups[g][0].replace('p_',''),metric.compute(predictions=pred, references=target)),file=f)
                else:
                    for g in range(ngroups):
                        pred = [item[g] for item in self.results['pred_str']]
                        target = [item[g] for item in self.results['target_text']]
                        print("Test group {} AER: {:.5f}".format(self.groups[g][0].replace('p_',''),metric.compute(predictions=pred, references=target)),file=f)
        


def main():
    fire.Fire(TrainSAModel)

if __name__ == '__main__':
    main()
