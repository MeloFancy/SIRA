# SIRA: a Semantic-aware, fIne-grained app Review Analysis approach
Dataset and replication package for the paper Where is Your App Frustrating Users? (ICSE 2022).

## Requirements
- Python 3.6
- Main libraries: requirements.txt
- [Pre-trained BERT](https://huggingface.co/bert-base-uncased)
	- Path: pytorch_version/prev_trained_model/

## Data
### Data Directory
The directory `dataset` is for the convenience of viewing the data, which contains labeled data and unlabeled data, respectively. \
The data directory when running our code: `pytorch_version/CLUEdatasets/cluener/`

### Data Field Format
```angular2html
app: App name
text: Review sentence
senti: Sentence sentiment (negative: [-5, -1], positive: [1, 5])
label: The problematic feature phrase, the beginning position and the ending position
```

## Usage
### Problematic Feature Extraction
```angular2html
run_ner_crf.sh
```

You can change the configures in `run_ner_crf.sh`, including the `learning_rate`, `per_gpu_train_batch_size`, `per_gpu_eval_batch_size`, `num_train_epochs`, etc. The other important parameters are
```angular2html
overwrite_output_dir -- whether overwrite the output directory
do_train -- whether train the model
do_eval -- whether evluate the model
do_predict -- whether predict the results of new data
```

### Problematic Feature Clustering
This should be run after obtaining the results of Problematic Feature Extraction.
Need revision based on the format of your output data, and assign your data to the variable `domain_docs`.
```angular2html
preprocess/clustering.py
```

## Output Directory
```angular2html
pytorch_version/outputs/cluener_output/bert/
```

## Visualization
[Tool Link](http://8.134.39.104:7490/SIRA/index.html)
