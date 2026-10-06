#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import fire
import yaml
import logging
from os.path import join
from os import makedirs
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union, Tuple

import numpy as np
import torch

from datasets import load_from_disk, concatenate_datasets, Dataset
import evaluate
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
_console = logging.StreamHandler()
_console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_console)


# -----------------------
# Data collator (CTC)
# -----------------------
@dataclass
class DataCollatorCTCWithPadding:
    processor: Wav2Vec2Processor
    padding_features: Union[bool, str] = True
    padding_labels: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding_features,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding_labels,
            max_length=self.max_length_labels,
            pad_to_multiple_of=self.pad_to_multiple_of_labels,
            return_tensors="pt",
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def _read_csv_first_column(csv_path: str) -> Tuple[List[str], List[str]]:
    """
    Robust CSV reader that returns:
      - header (list of column names, may be empty)
      - first_col_values (unique, order preserved)
    """
    import csv
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    seen = set()
    header: List[str] = []
    values: List[str] = []

    with open(csv_path, "r", encoding="utf-8") as f:
        # sniff delimiter (comma vs tab etc.)
        sample = f.read(4096)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";"])
        except Exception:
            dialect = csv.get_dialect("excel")

        reader = csv.reader(f, dialect)
        rows = list(reader)

    if not rows:
        return header, values

    # detect header: if first row has any non-phoneme-like strings, still accept as header
    # easiest: treat first row as header if it contains any alphabetic column-name-ish tokens
    # (works for "phoneme,attr1,attr2,...")
    first = [x.strip() for x in rows[0]]
    if any(x.lower() in ("phoneme", "phone", "token", "symbol") for x in first):
        header = first
        data_rows = rows[1:]
    else:
        data_rows = rows

    for r in data_rows:
        if not r:
            continue
        p = str(r[0]).strip()
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            values.append(p)

    return header, values


class TrainPhonemeCTC:
    """
    Phoneme-level only Wav2Vec2-CTC fine-tuning.
    YAML compatible with your config structure (datasets/phonological/preprocessor/training/evaluation/output).
    """

    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        # output
        self.working_dir = self.cfg["output"]["working_dir"]
        makedirs(self.working_dir, exist_ok=True)

        fh = logging.FileHandler(join(self.working_dir, "train_phoneme_ctc.log"))
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(fh)

        # datasets
        dcfg = self.cfg["datasets"]
        self.dataset_path = dcfg["data_path"]
        self.train_part = [x.strip() for x in str(dcfg["train_part"]).split(",")]
        self.validation_part = [x.strip() for x in str(dcfg["validation_part"]).split(",")]
        self.test_part = [x.strip() for x in str(dcfg.get("test_part", "")).split(",") if x.strip()]
        self.cache_dir = dcfg.get("cache_dir", None)

        # phonological (we only reuse phoneme2att_map_file as phoneme inventory source)
        phcfg = self.cfg.get("phonological", {})
        self.phoneme2att_map_file = phcfg.get("phoneme2att_map_file", "")
        # attribute_list_file / phonetic_alphabet are ignored for training logic (kept in yaml for compatibility)

        # preprocessor
        pcfg = self.cfg["preprocessor"]
        self.sampling_rate = int(pcfg.get("sampling_rate", 16000))
        self.do_normalize = bool(pcfg.get("do_normalize", True))
        self.return_attention_mask = bool(pcfg.get("return_attention_mask", False))
        self.phoneme_column = pcfg["phoneme_column"]  # transcript_IPAwithTone
        self.num_proc = int(pcfg.get("num_proc", 1))
        self.max_length_in_sec = float(pcfg.get("max_length_in_sec", 15))
        self.save_preprocessed_data = bool(pcfg.get("save_preprocessed_data", False))
        self.load_from_preprocessed_data = bool(pcfg.get("load_from_preprocessed_data", False))

        self.decouple_diphthongs = bool(pcfg.get("decouple_diphthongs", False))
        self.diph2mono_file = pcfg.get("diphthongs_to_monophthongs_map_file", "")

        # training
        tcfg = self.cfg["training"]
        self.model_path = tcfg["model_path"]  # local path you gave
        self.gradient_checkpointing = bool(tcfg.get("gradient_checkpointing", False))
        self.ctc_loss_reduction = str(tcfg.get("ctc_loss_reduction", "mean"))
        self.freeze_feature_encoder = bool(tcfg.get("freeze_feature_encoder", True))
        self.group_by_length = bool(tcfg.get("group_by_length", True))

        self.train_batch_size = int(tcfg.get("train_batch_size", 8))
        self.eval_batch_size = int(tcfg.get("eval_batch_size", self.train_batch_size))
        self.evaluation_strategy = str(tcfg.get("evaluation_strategy", "steps"))
        self.enable_fp16 = bool(tcfg.get("enable_fp16", False))
        self.num_train_epochs = float(tcfg.get("num_train_epochs", 10))
        self.save_steps = int(tcfg.get("save_steps", 500))
        self.eval_steps = int(tcfg.get("eval_steps", self.save_steps))
        self.logging_steps = int(tcfg.get("logging_steps", 100))
        self.prediction_loss_only = bool(tcfg.get("prediction_loss_only", False))
        self.learning_rate = float(tcfg.get("learning_rate", 1e-4))
        self.weight_decay = float(tcfg.get("weight_decay", 0.0))
        self.warmup_ratio = float(tcfg.get("warmup_ratio", 0.0))
        self.load_best_model_at_end = bool(tcfg.get("load_best_model_at_end", True))
        self.save_total_limit = int(tcfg.get("save_total_limit", 2))

        # evaluation
        ecfg = self.cfg.get("evaluation", {})
        self.metric_path = ecfg.get("metric_path", "")  # 'metrics/wer.py' (optional)
        self.auto_eval = bool(ecfg.get("auto_eval", False))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # internal
        self.diph2mono = None
        self.processor: Optional[Wav2Vec2Processor] = None
        self.model: Optional[Wav2Vec2ForCTC] = None
        self.data_train: Optional[Dataset] = None
        self.data_valid: Optional[Dataset] = None

    # -----------------------
    # phoneme inventory
    # -----------------------
    def _load_phoneme_inventory(self) -> List[str]:
        """
        Preferred: load phoneme set from phoneme2att_map_file first column.
        This matches your current yaml without adding new fields.
        """
        if not self.phoneme2att_map_file:
            raise ValueError("phonological.phoneme2att_map_file is empty; need it to build phoneme vocab.")
        _, phonemes = _read_csv_first_column(self.phoneme2att_map_file)
        if not phonemes:
            raise ValueError(f"No phonemes found in first column of {self.phoneme2att_map_file}")
        return phonemes

    def create_processor(self):
        phonemes = self._load_phoneme_inventory()

        # CTC vocab: <pad>=0, <unk>=1, then phonemes
        vocab = {"<pad>": 0, "<unk>": 1}
        for p in phonemes:
            if p in vocab:
                continue
            vocab[p] = len(vocab)

        vocab_file = join(self.working_dir, "phoneme_vocab.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file,
            pad_token="<pad>",
            unk_token="<unk>",
            word_delimiter_token="",  # we already split by spaces
        )
        feat = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            do_normalize=self.do_normalize,
            return_attention_mask=self.return_attention_mask,
        )
        self.processor = Wav2Vec2Processor(feature_extractor=feat, tokenizer=tokenizer)
        logger.info(f"Built tokenizer vocab_size={self.processor.tokenizer.vocab_size} from {self.phoneme2att_map_file}")

    # -----------------------
    # diphthong decouple (optional)
    # -----------------------
    def _load_diph2mono(self):
        """
        Your file: data/Diphthongs_Mandarin_withTone.csv
        We assume: first column = diph token, remaining columns = replacement tokens (mono1, mono2, ...)
        """
        if not self.diph2mono_file or (not os.path.exists(self.diph2mono_file)):
            raise FileNotFoundError(f"diphthongs_to_monophthongs_map_file not found: {self.diph2mono_file}")

        import csv
        mapping = {}
        with open(self.diph2mono_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                src = str(row[0]).strip()
                if not src:
                    continue
                dst_tokens = [str(x).strip() for x in row[1:] if str(x).strip()]
                if not dst_tokens:
                    continue
                mapping[src] = " ".join(dst_tokens)
        self.diph2mono = mapping
        logger.info(f"Loaded diph2mono mapping: {len(mapping)} entries from {self.diph2mono_file}")

    def _decouple_diphthongs(self, ex):
        tokens = str(ex[self.phoneme_column]).split()
        out = []
        for t in tokens:
            out.append(self.diph2mono.get(t, t))

        # optional: merge standalone tone digits back (rare but safe)
        merged = []
        for t in out:
            if t.isdigit():
                if merged:
                    merged[-1] = merged[-1] + t
                else:
                    merged.append(t)
            else:
                merged.append(t)

        ex[self.phoneme_column] = " ".join(merged)
        return ex

    # -----------------------
    # dataset preprocess
    # -----------------------
    def _prepare_batch(self, batch):
        # audio -> input_values
        rates = set([a["sampling_rate"] for a in batch["audio"]])
        if len(rates) != 1 or list(rates)[0] != self.sampling_rate:
            raise ValueError(f"Sampling rate mismatch: got {rates}, expected {self.sampling_rate}")

        batch["input_values"] = self.processor(
            audio=[a["array"] for a in batch["audio"]],
            sampling_rate=self.sampling_rate
        ).input_values

        # labels: split phoneme string into tokens; use is_split_into_words=True to avoid substring tokenization bugs
        ph_tokens = [str(x).split() for x in batch[self.phoneme_column]]
        batch["labels"] = self.processor.tokenizer(ph_tokens, is_split_into_words=True).input_ids
        return batch

    def preprocess_data(self, ds: Dataset) -> Dataset:
        # filter empty transcript
        ds = ds.filter(lambda x: str(x[self.phoneme_column]).strip() != "", num_proc=self.num_proc)

        # filter duration
        ds = ds.filter(
            lambda x: len(x["audio"]["array"]) < self.max_length_in_sec * x["audio"]["sampling_rate"],
            num_proc=self.num_proc,
        )

        if self.decouple_diphthongs:
            if self.diph2mono is None:
                self._load_diph2mono()
            ds = ds.map(self._decouple_diphthongs, batched=False)

        # to input_values + labels
        ds = ds.map(
            self._prepare_batch,
            batched=True,
            batch_size=8,
            num_proc=self.num_proc,
            remove_columns=ds.column_names,
        )
        return ds

    def load_data(self):
        prep_root = join(self.working_dir, "preprocessed_data_phoneme")

        if self.load_from_preprocessed_data:
            try:
                self.data_train = load_from_disk(join(prep_root, "train"))
                self.data_valid = load_from_disk(join(prep_root, "valid"))
                logger.info("Loaded preprocessed datasets from disk.")
                return
            except Exception as e:
                logger.warning(f"Failed to load preprocessed data: {e}. Reprocessing from raw.")

        data = load_from_disk(self.dataset_path)

        train_ds = concatenate_datasets([data[k] for k in self.train_part])
        valid_ds = concatenate_datasets([data[k] for k in self.validation_part])

        self.data_train = self.preprocess_data(train_ds)
        self.data_valid = self.preprocess_data(valid_ds)

        if self.save_preprocessed_data:
            makedirs(prep_root, exist_ok=True)
            self.data_train.save_to_disk(join(prep_root, "train"))
            self.data_valid.save_to_disk(join(prep_root, "valid"))
            logger.info(f"Saved preprocessed datasets to {prep_root}")

    # -----------------------
    # metrics: PER
    # -----------------------


    def build_metrics(self):
        tok = self.processor.tokenizer
        blank_id = tok.pad_token_id  # Wav2Vec2 CTC blank is pad_token_id
        
        def _ctc_collapse_and_remove_blanks(ids, blank_id: int):
            out = []
            prev = None
            for i in ids:
                if i == prev:
                    continue
                prev = i
                if i == blank_id:
                    continue
                out.append(i)
            return out

        def _edit_distance(a, b):
            # token-level Levenshtein
            n, m = len(a), len(b)
            dp = list(range(m + 1))
            for i in range(1, n + 1):
                prev = dp[0]
                dp[0] = i
                for j in range(1, m + 1):
                    cur = dp[j]
                    cost = 0 if a[i - 1] == b[j - 1] else 1
                    dp[j] = min(
                        dp[j] + 1,      # deletion
                        dp[j - 1] + 1,  # insertion
                        prev + cost     # substitution
                    )
                    prev = cur
            return dp[m]

        def compute_metrics(pred):
            # pred.predictions: (B, T, V)
            pred_ids = np.argmax(pred.predictions, axis=-1)

            # labels: (B, L) with -100 padding
            label_ids = pred.label_ids.copy()
            label_ids[label_ids == -100] = blank_id

            total_dist = 0
            total_ref = 0

            for p_seq, r_seq in zip(pred_ids, label_ids):
                p_seq = _ctc_collapse_and_remove_blanks(p_seq.tolist(), blank_id)
                r_seq = [i for i in r_seq.tolist() if i != blank_id]

                # (可选) 过滤掉 <unk>，看你要不要把 unk 当 error
                # unk_id = tok.unk_token_id
                # p_seq = [i for i in p_seq if i != unk_id]
                # r_seq = [i for i in r_seq if i != unk_id]

                dist = _edit_distance(p_seq, r_seq)
                total_dist += dist
                total_ref += max(1, len(r_seq))

            per = total_dist / total_ref
            return {"per": per}

        self.compute_metrics = compute_metrics

    # -----------------------
    # trainer
    # -----------------------
    def prepare_trainer(self):
        assert self.processor is not None, "processor not built"

        self.model = Wav2Vec2ForCTC.from_pretrained(
            self.model_path,
            local_files_only=True,  # your model is local
            gradient_checkpointing=self.gradient_checkpointing,
            ctc_loss_reduction=self.ctc_loss_reduction,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=self.processor.tokenizer.vocab_size,
            cache_dir=self.cache_dir,
        )
        self.model.config.ctc_zero_infinity = True

        if self.freeze_feature_encoder:
            self.model.freeze_feature_encoder()

        self.model.to(self.device)

        training_args = TrainingArguments(
            output_dir=join(self.working_dir, "fine_tune_phoneme"),
            group_by_length=self.group_by_length,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            evaluation_strategy=self.evaluation_strategy,
            fp16=self.enable_fp16,
            num_train_epochs=self.num_train_epochs,
            save_steps=self.save_steps,
            logging_steps=self.logging_steps,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_ratio=self.warmup_ratio,
            load_best_model_at_end=self.load_best_model_at_end,
            save_total_limit=self.save_total_limit,
            metric_for_best_model="per",
            greater_is_better=False,
            prediction_loss_only=self.prediction_loss_only,
            report_to="none",
            eval_steps=self.eval_steps,
            
        )

        data_collator = DataCollatorCTCWithPadding(self.processor)

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=self.data_train,
            eval_dataset=self.data_valid,
            tokenizer=self.processor.feature_extractor,
            compute_metrics=None if self.prediction_loss_only else self.compute_metrics,
        )

    def save_best(self):
        out_dir = join(self.working_dir, "fine_tune_phoneme", "best")
        makedirs(out_dir, exist_ok=True)
        self.trainer.save_model(out_dir)
        self.processor.save_pretrained(out_dir)
        logger.info(f"Saved best model to: {out_dir}")

    # -----------------------
    # entry
    # -----------------------
    def train(self, resume_from_checkpoint: Optional[str] = None):
        torch.cuda.empty_cache()
        self.create_processor()
        self.load_data()
        self.build_metrics()
        self.prepare_trainer()

        if resume_from_checkpoint:
            logger.info(f"Resuming from checkpoint: {resume_from_checkpoint}")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()

        self.save_best()


def main():
    fire.Fire(TrainPhonemeCTC)

if __name__ == "__main__":
    main()

