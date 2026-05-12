# Speech Attribute Transcription

Fine-tune self-supervised speech foundation models such as wav2vec 2.0, HuBERT, and WavLM to recognize phonological or speech attributes from audio, then optionally reconstruct phoneme sequences from the predicted attribute streams.

The project is designed for multilingual and custom attribute experiments. If your dataset contains speech audio and a phonetic transcription, and you can provide a CSV mapping from phonemes to binary attributes, the same training pipeline can be used for another language, alphabet, or attribute inventory.

## What This Repository Does

Most CTC speech recognizers predict phonemes or characters directly. This repository trains a model to predict binary speech-attribute sequences instead:

```text
phoneme transcript:  dh ah k ae t
attribute target:    +fricative -fricative -fricative -fricative -fricative
attribute target:    -vowel     +vowel     -vowel     +vowel     -vowel
...
```

For each attribute in the configured attribute list, `train.py` creates two CTC tokens:

```text
p_<attribute>   positive value, CSV value 1
n_<attribute>   negative value, CSV value 0
```

The model uses a shared output vocabulary, but the custom trainer computes a separate binary CTC loss for every attribute group. At inference time, `transcriber.py` can decode either the attribute streams or a phoneme sequence by comparing the predicted attributes against a phoneme-to-attribute matrix.

## Repository Layout

```text
train.py             Train and evaluate speech-attribute models
train_phModel.py     Train and evaluate a direct phoneme-recognition CTC model
transcriber.py       Run inference with a trained speech-attribute model
Phonemize.py         Optional helper to add phoneme columns to text datasets
configs/            Example YAML experiment configurations
data/               Example phoneme-to-attribute CSVs and attribute lists
metrics/wer.py      WER/PER/AER metric used by evaluation scripts
metrics/cm.py       Phoneme confusion matrix helper
```

## Installation

Python 3.10 or later is recommended. Install PyTorch for your CUDA or CPU environment first if you need a specific build, then install the repository requirements.

```bash
git clone https://github.com/<your-org>/Speech-Attribute-Transcription.git
cd Speech-Attribute-Transcription

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The training scripts currently support a single GPU. If your machine has multiple GPUs, select one explicitly:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --config_file=configs/test_timit.yaml train_SA_model
```

## Data Format

Training data is loaded with Hugging Face `datasets.load_from_disk`, so `datasets.data_path` must point to a saved `Dataset` or `DatasetDict`.

Each example must contain:

- `audio`: a Hugging Face audio column or dictionary with `array` and `sampling_rate`
- a phoneme transcript column, configured by `preprocessor.phoneme_column`

The phoneme transcript must be a whitespace-separated sequence whose symbols match the configured phoneme-to-attribute CSV:

```text
dh ah k ae t
```

Example dataset structure:

```text
DatasetDict({
  train: Dataset({
    features: ["audio", "phoneme", ...]
  })
  validation: Dataset({
    features: ["audio", "phoneme", ...]
  })
  test: Dataset({
    features: ["audio", "phoneme", ...]
  })
})
```

The code expects 16 kHz audio during preprocessing. Resample your dataset before saving it if needed.

## Phoneme-to-Attribute Mapping

The mapping CSV defines the phoneme inventory and the binary attributes for each phoneme. See `data/Phoneme2att_ipa_att_Diph.csv` for an example.

The training script expects:

- one phoneme column named `Phoneme_<phonetic_alphabet>`
- one column for each attribute listed in `phonological.attribute_list_file`
- attribute values encoded as `0` or `1`

Example:

```csv
Phoneme_arpa,Phoneme_ipa,Language,bilabial,plosive,voiced,vowel,fricative
b,b,en_us,1,1,1,0,0
p,p,en_us,1,1,0,0,0
aa,a,en_us,0,0,0,1,0
```

The attribute list file is a plain text file with one attribute name per line:

```text
bilabial
plosive
voiced
vowel
fricative
```

The `phonological.phonetic_alphabet` field selects which phoneme column to use. For example:

```yaml
phonological:
  phoneme2att_map_file: data/Phoneme2att_ipa_att_Diph.csv
  attribute_list_file: data/list_attributes-ipa_Diph.txt
  phonetic_alphabet: arpa
```

This makes training read phoneme symbols from the `Phoneme_arpa` column.

For phoneme reconstruction in `transcriber.py`, the first column of the CSV is used as the output phoneme inventory. Use a mapping file whose first column contains the symbols you want the transcriber to output.

## Configuration

Experiments are controlled by YAML files in `configs/`. A typical configuration has five sections.

```yaml
datasets:
  data_path: /path/to/saved_dataset
  train_part: train
  validation_part: validation
  test_part: test
  cache_dir: /path/to/hf_cache

phonological:
  attribute_list_file: data/list_attributes-ipa_Diph.txt
  phoneme2att_map_file: data/Phoneme2att_ipa_att_Diph.csv
  phonetic_alphabet: arpa

preprocessor:
  sampling_rate: 16000
  do_normalize: true
  return_attention_mask: false
  phoneme_column: phoneme
  do_phonemize: false
  num_proc: 1
  max_length_in_sec: 15
  min_length_in_sec: 0.1
  save_preprocessed_data: true
  load_from_preprocessed_data: true
  decouple_diphthongs: false
  diphthongs_to_monophthongs_map_file: data/Diphthongs_en_us-arpa.csv

training:
  model_path: facebook/wav2vec2-large-xlsr-53
  model_type: WAV2VEC2
  gradient_checkpointing: true
  ctc_loss_reduction: mean
  freeze_feature_encoder: true
  group_by_length: true
  train_batch_size: 32
  evaluation_strategy: steps
  enable_fp16: true
  num_train_epochs: 10
  save_steps: 100
  logging_steps: 100
  prediction_loss_only: true
  learning_rate: 1e-4
  weight_decay: 0.005
  warmup_ratio: 0.1
  load_best_model_at_end: true
  save_total_limit: 3

evaluation:
  spaces_between_special_tokens: true
  metric_path: metrics/wer.py
  eval_extra_data: ""
  eval_extra_data_parts: ""
  auto_eval: true

output:
  working_dir: working/my_experiment
```

Supported `training.model_type` values are:

- `WAV2VEC2`
- `HuBERT`
- `WavLM`

`training.model_path` can be a local model directory or a Hugging Face model name compatible with the selected model type.

## Train a Speech-Attribute Model

Edit one of the example configs or create a new YAML file, then run:

```bash
python train.py --config_file=configs/test_timit.yaml train_SA_model
```

Resume from a checkpoint:

```bash
python train.py --config_file=configs/test_timit.yaml train_SA_model --resume_from_checkpoint=true
```

Override the model type from the command line:

```bash
python train.py --config_file=configs/test_timit_HuBERT.yaml train_SA_model --model_type=HuBERT
```

Outputs are written under `output.working_dir`:

```text
working/my_experiment/
  log
  vocab.json
  preprocessed_data/
  fine_tune/
    checkpoint-...
    best/
      config.json
      model.safetensors or pytorch_model.bin
      preprocessor_config.json
      tokenizer_config.json
      vocab.json
```

The saved model in `fine_tune/best` contains both the acoustic model and processor needed for inference.

## Evaluate a Speech-Attribute Model

If `evaluation.auto_eval` is `true`, evaluation runs automatically after training.

To evaluate a trained model manually using the dataset configured in YAML:

```bash
python train.py --config_file=configs/test_timit.yaml evaluate_SA_model
```

To evaluate another saved dataset or split:

```bash
python train.py \
  --config_file=configs/test_timit.yaml \
  evaluate_SA_model \
  --eval_data=/path/to/saved_dataset \
  --eval_parts=test \
  --suffix=timit \
  --phoneme_column=phoneme
```

Evaluation writes:

```text
working/my_experiment/results_<suffix>.db
working/my_experiment/results_<suffix>.txt
```

The text report contains attribute error rate (AER) for each configured attribute.

## Inference

Use `transcriber.py` with the saved speech-attribute model.

List the attributes supported by a model:

```bash
python transcriber.py \
  --model_path=working/my_experiment/fine_tune/best \
  --model_type=WAV2VEC2 \
  print_availabel_attributes
```

Transcribe a single audio file and print all attribute streams:

```bash
python transcriber.py \
  --model_path=working/my_experiment/fine_tune/best \
  --model_type=WAV2VEC2 \
  transcribe \
  --audio=/path/to/audio.wav \
  --attributes=all
```

Transcribe selected attributes:

```bash
python transcriber.py \
  --model_path=working/my_experiment/fine_tune/best \
  --model_type=WAV2VEC2 \
  transcribe \
  --audio=/path/to/audio.wav \
  --attributes="('vowel','fricative','voiced')"
```

When passing more than one attribute through the command line, use a Python tuple literal as shown above. Programmatically, pass a tuple such as `("vowel", "fricative", "voiced")`.

Decode phonemes from predicted attributes by providing a phoneme-to-attribute matrix:

```bash
python transcriber.py \
  --model_path=working/my_experiment/fine_tune/best \
  --model_type=WAV2VEC2 \
  transcribe \
  --audio=/path/to/audio.wav \
  --attributes=all \
  --phonological_matrix_file=data/Phoneme2att_ipa_att_Diph.csv
```

Batch transcribe a saved Hugging Face dataset:

```bash
python transcriber.py \
  --model_path=working/my_experiment/fine_tune/best \
  --model_type=WAV2VEC2 \
  transcribe_dataset \
  --input_dataset_path=/path/to/saved_dataset \
  --output_dataset_path=working/my_experiment/predicted_dataset \
  --split=test \
  --phonological_matrix_file=data/Phoneme2att_ipa_att_Diph.csv \
  --recognize_phoneme=true
```

Evaluate a predicted dataset:

```bash
python transcriber.py \
  --model_path=working/my_experiment/fine_tune/best \
  --model_type=WAV2VEC2 \
  evaluate_dataset \
  --input_dataset_path=working/my_experiment/predicted_dataset \
  --split=test \
  --pred_phoneme=pred_phoneme \
  --ref_phoneme=phoneme \
  --metric_path=metrics/wer.py
```

## Train a Direct Phoneme Model

`train_phModel.py` trains a standard CTC phoneme recognizer without speech attributes. This is useful as a baseline against the attribute-based model.

The config format is similar, but it uses:

```yaml
datasets:
  phoneme_list_file: data/list_Phoneme_Arpa_noDiph.txt
```

Add `phoneme_list_file` to a phoneme-model YAML config, then run training:

```bash
python train_phModel.py --config_file=configs/my_phoneme_model.yaml train_model
```

Run evaluation:

```bash
python train_phModel.py --config_file=configs/my_phoneme_model.yaml evaluate_model
```

The phoneme model writes its best checkpoint to:

```text
<working_dir>/fine_tune/best
```

## Preparing Phoneme Transcriptions

If your dataset has text but no phoneme column, `Phonemize.py` can add phoneme transcriptions using supported English G2P backends:

```bash
python Phonemize.py run \
  --dataset_path=/path/to/text_audio_dataset \
  --output_path=/path/to/phonemized_dataset \
  --phonemizers=cmu \
  --normalize=true \
  --nproc=4
```

For other languages, prepare phoneme transcriptions with your preferred G2P or forced alignment tool, then save them as a whitespace-separated phoneme column in the Hugging Face dataset.

## Adapting to a New Language or Attribute Set

1. Prepare a Hugging Face `DatasetDict` with 16 kHz audio and phonetic transcriptions.
2. Decide the phoneme alphabet used in the transcript, such as IPA, ARPAbet, X-SAMPA, or a language-specific symbol set.
3. Create a phoneme-to-attribute CSV with one row per phoneme and one binary column per attribute.
4. Create an attribute list text file containing the exact attribute column names to train.
5. Set `phonological.phonetic_alphabet` so it matches the CSV phoneme column name `Phoneme_<phonetic_alphabet>`.
6. Set `preprocessor.phoneme_column` to the dataset column containing phoneme sequences.
7. Choose a compatible foundation model and set `training.model_path` and `training.model_type`.
8. Train with `train.py`, then use `transcriber.py` to decode attributes or phonemes.

Important checks:

- Every phoneme appearing in the dataset transcript must appear in the selected `Phoneme_<phonetic_alphabet>` CSV column.
- Every attribute in the attribute list must exist as a CSV column.
- Attribute values must be binary, with `1` for positive and `0` for negative.
- Attribute names should be simple token-safe strings, for example `vowel`, `nasal`, or `labiodental`.
- If you decouple diphthongs, provide a diphthong-to-monophthong CSV such as `data/Diphthongs_en_us-arpa.csv`.
- When changing the phoneme column, attribute list, or mapping CSV, use a new `output.working_dir` or disable `load_from_preprocessed_data` so old cached targets are not reused.

## Notes and Limitations

- The current code asserts 16 kHz input audio during preprocessing.
- Training is implemented for a single GPU. Use `CUDA_VISIBLE_DEVICES` to choose the GPU on multi-GPU systems.
- The speech-attribute model predicts attributes independently through grouped binary CTC losses. Phoneme reconstruction depends on how uniquely the chosen attributes identify the phoneme inventory.
- If two phonemes have identical attribute vectors, the transcriber cannot reliably distinguish them from attributes alone.
- `do_phonemize` is present in the YAML configs, but phonemization is handled separately by `Phonemize.py`.

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{shahin2025phonological,
  title={Phonological level wav2vec2-based Mispronunciation Detection and Diagnosis method},
  author={Shahin, Mostafa and Epps, Julien and Ahmed, Beena},
  journal={Speech Communication},
  volume={173},
  pages={103249},
  year={2025},
  publisher={Elsevier}
}
```
