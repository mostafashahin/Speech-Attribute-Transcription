# Speech-Attribute-Transcription
## Training
```
python3 train.py --config_file=config_libri_100_10epoch.yaml train_SA_model
```
## Evaluation
```
python3 train.py --config_file="config_libri_100_10epoch_sb.yaml" evaluate_SA_model --eval_data="../datasets/timit/" --eval_parts="test" --suffix="timit" --phoneme_column="phoneme_dp=>> PBS: job killed: walltime 43298 exceeded limit 43200
```
