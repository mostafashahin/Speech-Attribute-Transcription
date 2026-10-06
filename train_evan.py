#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import yaml
import fire
import logging
from dataclasses import dataclass
from os import makedirs
from os.path import join
from typing import Optional, Union, List, Dict

import pandas as pd
import torch
from datasets import load_from_disk, DatasetDict, Dataset, concatenate_datasets

from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    Trainer,
    TrainingArguments,
    DataCollatorCTCWithPadding,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


def _safe_split_tokens(s: str) -> List[str]:
    # 统一空格分 token（确保你的 phoneme 序列是 "p a2 i4" 这种）
    s = (s or "").strip()
    return s.split() if s else []


class TrainPhonemeCTC:
    """
    训练一个 phoneme-level CTC baseline（对齐 Shahin 的 phoneme acoustic model）
    """

    def __init__(self, config_file: str):
        with open(config_file, "r") as f:
            self.cfg = yaml.safe_load(f)

        # device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.n_devices = torch.cuda.device_count()
            assert self.n_devices == 1, "仅支持单卡。多卡请用 CUDA_VISIBLE_DEVICES=0"
        else:
            self.device = torch.device("cpu")
            self.n_devices = 1

        # paths / dataset
        self.working_dir = self.cfg["output"]["working_dir"]
        makedirs(self.working_dir, exist_ok=True)

        self.dataset_path = self.cfg["datasets"]["data_path"]
        self.cache_dir = self.cfg["datasets"]["cache_dir"]
        self.train_part = [x.strip() for x in self.cfg["datasets"]["train_part"].split(",")]
        self.validation_part = [x.strip() for x in self.cfg["datasets"]["validation_part"].split(",")]
        self.test_part = [x.strip() for x in self.cfg["datasets"]["test_part"].split(",")]

        # phoneme label column in dataset (e.g., transcript_ipa / IPA / IPAtone)
        self.phoneme_column = self.cfg["preprocessor"]["phoneme_column"]
        self.sampling_rate = self.cfg["preprocessor"]["sampling_rate"]
        self.do_normalize = self.cfg["preprocessor"]["do_normalize"]
        self.return_attention_mask = self.cfg["preprocessor"]["return_attention_mask"]
        self.num_proc = self.cfg["preprocessor"]["num_proc"]
        self.max_length_in_sec = self.cfg["preprocessor"]["max_length_in_sec"]
        self.min_length_in_sec = self.cfg["preprocessor"].get("min_length_in_sec", 0.1)

        # IMPORTANT: 这里用你的 phoneme2att_map 只是为了拿 phoneme inventory 生成 vocab
        self.phoneme2att_map_file = self.cfg["phonological"]["phoneme2att_map_file"]
        self.phonetic_alphabet = self.cfg["phonological"]["phonetic_alphabet"]  # e.g., "ipaDragon"

        # training config
        self.model_path = self.cfg["training"]["model_path"]
        self.gradient_checkpointing = self.cfg["training"]["gradient_checkpointing"]
        self.ctc_loss_reduction = self.cfg["training"]["ctc_loss_reduction"]
        self.freeze_feature_encoder = self.cfg["training"]["freeze_feature_encoder"]
        self.group_by_length = self.cfg["training"]["group_by_length"]
        self.train_batch_size = self.cfg["training"]["train_batch_size"]
        self.evaluation_strategy = self.cfg["training"]["evaluation_strategy"]
        self.enable_fp16 = self.cfg["training"]["enable_fp16"]
        self.num_train_epochs = self.cfg["training"]["num_train_epochs"]
        self.save_steps = self.cfg["training"]["save_steps"]
        self.logging_steps = self.cfg["training"]["logging_steps"]
        self.prediction_loss_only = self.cfg["training"]["prediction_loss_only"]
        self.learning_rate = float(self.cfg["training"]["learning_rate"])
        self.weight_decay = self.cfg["training"]["weight_decay"]
        self.warmup_ratio = self.cfg["training"]["warmup_ratio"]
        self.load_best_model_at_end = self.cfg["training"]["load_best_model_at_end"]
        self.save_total_limit = self.cfg["training"]["save_total_limit"]

        self.saved_model_path = join(self.working_dir, "fine_tune_phoneme", "best")

    # ---------- vocab / processor ----------
    def load_phoneme_inventory(self) -> List[str]:
        df = pd.read_csv(self.phoneme2att_map_file)
        col = f"Phoneme_{self.phonetic_alphabet}"
        assert col in df.columns, f"列不存在: {col} (检查 phonetic_alphabet / csv header)"

        phs = [str(x).strip() for x in df[col].tolist() if str(x).strip() != ""]
        # 去重保持顺序
        seen = set()
        uniq = []
        for p in phs:
            if p not in seen:
                seen.add(p)
                uniq.append(p)

        logger.info(f"Loaded phoneme inventory size = {len(uniq)} from {col}")
        return uniq

    def create_processor(self):
        phonemes = self.load_phoneme_inventory()

        vocab = {p: i + 2 for i, p in enumerate(phonemes)}
        vocab["<pad>"] = 0
        vocab["<unk>"] = 1
        vocab = dict(sorted(vocab.items(), key=lambda x: x[1]))

        vocab_file = join(self.working_dir, "vocab_phoneme.json")
        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        tokenizer = Wav2Vec2CTCTokenizer(
            vocab_file,
            pad_token="<pad>",
            unk_token="<unk>",
            word_delimiter_token="",  # 我们用空格切 token，但不需要 delimiter token
            do_lower_case=False,
        )
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.sampling_rate,
            padding_value=0.0,
            do_normalize=self.do_normalize,
            return_attention_mask=self.return_attention_mask,
        )
        self.processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

    # ---------- dataset ----------
    def _filter_and_basic(self, data: Dataset) -> Dataset:
        # 过滤空标签
        data = data.filter(lambda x: (x.get(self.phoneme_column, "") or "").strip() != "", num_proc=self.num_proc)

        # 过滤太短/太长音频
        def ok_len(x):
            n = len(x["audio"]["array"])
            sr = x["audio"]["sampling_rate"]
            return (n > self.min_length_in_sec * sr) and (n < self.max_length_in_sec * sr)

        data = data.filter(ok_len, num_proc=self.num_proc)
        return data

    def _prepare_dataset(self, batch):
        # 保证 SR 一致
        srs = set([a["sampling_rate"] for a in batch["audio"]])
        assert len(srs) == 1 and list(srs)[0] == self.sampling_rate, f"SR mismatch: {srs}"

        # audio -> input_values
        batch["input_values"] = self.processor(
            audio=[a["array"] for a in batch["audio"]],
            sampling_rate=batch["audio"][0]["sampling_rate"],
        ).input_values

        # phoneme string -> token ids
        # IMPORTANT: is_split_into_words=True 避免 tokenizer 把符号拆开
        token_lists = [_safe_split_tokens(s) for s in batch[self.phoneme_column]]
        with self.processor.as_target_processor():
            batch["labels"] = self.processor.tokenizer(
                token_lists, is_split_into_words=True
            ).input_ids

        return batch

    def load_data(self):
        data = load_from_disk(self.dataset_path)

        # train
        train_sets = [data[k] for k in self.train_part]
        self.data_train = concatenate_datasets(train_sets)
        self.data_train = self._filter_and_basic(self.data_train)
        self.data_train = self.data_train.map(
            self._prepare_dataset,
            batched=True,
            batch_size=8,
            num_proc=self.num_proc,
            remove_columns=self.data_train.column_names,
        )

        # valid
        valid_sets = [data[k] for k in self.validation_part]
        self.data_valid = concatenate_datasets(valid_sets)
        self.data_valid = self._filter_and_basic(self.data_valid)
        self.data_valid = self.data_valid.map(
            self._prepare_dataset,
            batched=True,
            batch_size=8,
            num_proc=self.num_proc,
            remove_columns=self.data_valid.column_names,
        )

        logger.info(f"Train size={len(self.data_train)}, Valid size={len(self.data_valid)}")

    # ---------- trainer ----------
    def prepare_trainer(self):
        collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

        try:
            model = Wav2Vec2ForCTC.from_pretrained(
                self.model_path,
                use_safetensors=True,
                gradient_checkpointing=self.gradient_checkpointing,
                ctc_loss_reduction=self.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
                cache_dir=self.cache_dir,
            )
        except Exception:
            logger.warning("Online load failed; try local_files_only...")
            model = Wav2Vec2ForCTC.from_pretrained(
                self.model_path,
                local_files_only=True,
                use_safetensors=False,
                gradient_checkpointing=self.gradient_checkpointing,
                ctc_loss_reduction=self.ctc_loss_reduction,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
                cache_dir=self.cache_dir,
            )

        model.config.ctc_zero_infinity = True
        if self.freeze_feature_encoder:
            model.freeze_feature_encoder()
        model.to(self.device)
        self.model = model

        args = TrainingArguments(
            output_dir=join(self.working_dir, "fine_tune_phoneme"),
            group_by_length=self.group_by_length,
            per_device_train_batch_size=int(self.train_batch_size / self.n_devices),
            eval_strategy=self.evaluation_strategy,
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
            max_grad_norm=10.0,
        )

        self.trainer = Trainer(
            model=self.model,
            args=args,
            data_collator=collator,
            train_dataset=self.data_train,
            eval_dataset=self.data_valid,
            tokenizer=self.processor.feature_extractor,
        )

    def save_model(self):
        makedirs(self.saved_model_path, exist_ok=True)
        self.model.save_pretrained(self.saved_model_path)
        self.processor.save_pretrained(self.saved_model_path)
        logger.info(f"Saved phoneme CTC model to: {self.saved_model_path}")

    # ---------- entry ----------
    def train(self, resume_from_checkpoint: Optional[str] = None):
        torch.cuda.empty_cache()
        self.create_processor()
        self.load_data()
        self.prepare_trainer()

        if resume_from_checkpoint:
            logger.info(f"Resume from: {resume_from_checkpoint}")
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            self.trainer.train()

        self.save_model()


def main():
    fire.Fire(TrainPhonemeCTC)


if __name__ == "__main__":
    main()

