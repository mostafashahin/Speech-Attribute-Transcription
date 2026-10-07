# Speech-Attribute-Transcription

# Jinghao: This Repo is not finished yet

## Training
```
python3 train.py --config_file=config_libri_100_10epoch.yaml train_SA_model

python train.py --config_file=mandarin_setting.yaml train_SA_model
```
## Evaluation
```
python3 train.py --config_file="config_libri_100_10epoch.yaml" evaluate_SA_model --eval_data="../datasets/timit/" --eval_parts="test" --suffix="timit" --phoneme_column="phoneme_dp"

python train.py --config_file="mandarin_setting.yaml" evaluate_SA_model --eval_data="C:\Users\Evanc\Desktop\Machine_Leraning\Mostafa_Phono_model\speech_attribute\datasets\CommonVoice_down16K" --eval_parts="test" --suffix="CVoice13" --phoneme_column="transcript"

python train.py --config_file="mandarin_setting.yaml" evaluate_SA_model --eval_data="C:\Users\Evanc\Desktop\modelResults\results_test_validation.db" --eval_parts="test" --suffix="timit" --phoneme_column="transcript"

pip install build cmake ninja wheel

```