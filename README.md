## A Schema-aware End-to-End model for Task-Oriented Dialogue Systems (SchemaTOD)
This is a repository for codes and data used for our paper, "End-to-End Task-Oriented Dialogue Systems Based on Schema", accepted to Findings of ACL 2023.

### Requirements
```python
pip install -r requirements.txt
```

### Data Preprocessing
We preprocessed data from [the Schema-Guided Dialogue dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue).
All preprocessed data are available under `data/sgd-processed`.
To run the data preprocessing, please follow:
```python
unzip data.zip
cd SchemaTOD
python preprocessing.py
```

### Data Format
The example turns in a processed dialogue from `data/sgd-processed/data.json`.
```json
{
    "user": "i'm looking for a place to eat.",
    "user_delex": "i'm looking for a place to eat.",
    "resp": "which city should i search in? what kind of food are you looking for?",
    "pointer": "1,0,0,0,0,0",
    "match": "0",
    "constraint": "",
    "cons_delex": "",
    "sys_act": "[restaurants] [request] [city] [cuisine]",
    "turn_num": 0,
    "turn_serv": "[restaurants_1]"
},
{
    "user": "some freshwater fish kind of food in palo alto would be perfect.",
    "user_delex": "some [cuisine] kind of food in [city] would be perfect.",
    "resp": "i found [count] restaurants. [restaurant_name] is a nice restaurant in [city].",
    "pointer": "0,0,0,1,0,0",
    "match": "4",
    "constraint": "[restaurants] [city] palo alto [cuisine] seafood",
    "cons_delex": "[restaurants] [city] [cuisine]",
    "sys_act": "[restaurants] [offer] [count] [restaurant_name] [city]",
    "turn_num": 1,
    "turn_serv": "[restaurants_1]"
},
```
- `user`: a user utterance
- `user_delex`: a delexicalized user utterance where a value is replaced by [slot]
- `resp`: a system response
- `pointer`: 1st-4th is a DB pointer specified a number of entities by a current constraint (1st: no entities, 2nd: one entity, 3rd less than or equal to 3 entities, 4th more than 3 entities), and 5th-6th is a notification pointer (5th: failure, 6th: success)
- `match`: an actual number of entities retrieved using a current constraint
- `constraint`: a current constraint (based on a user's goal) represented by [domain] [slot] slot_value
- `cons_delex`: a delexicalized constraint represented by [domain] [slot] [slot]
- `sys_act`: a system action with [domain] [action_type] [slot]
- `turn_num`: a number of turn
- `turn_serv`: services for the turn

### Training
To train SchemaTOD, 
```python
python train.py -mode train -cfg batch_size=16 epoch_num=15 exp_no=trial cuda=True
```
- `mode`: train or resume
- `model_path`: path to a pretrained model (e.g. t5-base)
- `tok_path`: path to a tokenizer (e.g. t5-base)
- `inference_path`: a path to a checkpoint of the latest trained epoch in case of resuming
- `cuda`: whether to use CPU or GPU, default=True
- `validate_during_training`: whether to validate after training of each epoch, default=True
- `report_interval`: a report frequency, default=4 
- `max_context_length`: a maximum number of the input context length, default=768
- `use_context`: whether to use schema context information, default=True
- `include_context_desc`: whether to include slot descriptions in the context, default=True
- `include_context_val_cons`: whether to include slot value constraints in the context, default=True
- `lr`: a learning rate, default=3e-4
- `gradient_accumulation_steps`: a gradient accumulatin step 
- `batch_size`: a batch size
- `epoch_num`: a total number of training epochs
- `exp_domains`: domains to be excluded, default=all
- `ft_domains`: domains to be finetuned, default=all

### Inference
For the generation,
```python
python train.py -mode test -cfg batch_size=16 inference_path=experiments/all_trial_bs16_ga1/epoch15 cuda=True
```
- `mode`: dev or test
- `inference_path`: a path to a checkpoint of the trained model
- `cuda`: whether to use CPU or GPU, default=True
- `max_context_length`: a maximum number of the input context length, default=768
- `use_context`: whether to use schema context information, default=True
- `include_context_desc`: whether to include slot descriptions in the context, default=True
- `include_context_val_cons`: whether to include slot value constraints in the context, default=True
- `batch_size`: a batch size
- `eval_domains`: domains to evaluate, default=all
- `use_true_prev_dspn`: whether to use true dspn for the dialogue history, default=False 
- `use_true_prev_bspn`: whether to use true bspn for the dialogue history, default=False
- `use_true_db_pointer`: whether to use true db pointer for the dialogue history, default=False
- `use_true_prev_aspn`: whether to use true aspn for the dialogue history, default=False 
- `use_true_prev_resp`: whether to use true resp for the dialogue history, default=False
- `use_true_curr_dspn`: whether to use true dspn for the current dialogue turn, default=False 
- `use_true_curr_bspn`: whether to use true bspn for the current dialogue turn, default=False 
- `use_all_previous_context`: whether to use all information in the dialogue history (i.e. bspn, dspn, db pointer, aspn, and resp) 
- `do_sample`: whether to use sampling method, default=False
- `temperature`: a value [0, 1] to decrease/increase the next token probabilities, default=0.7

### Acknowledgements
The code is implemented based on the released codes of the following papers:
- SOLOIST: Building Task Bots at Scale with Transfer Learning and Machine Teaching [[paper]](https://aclanthology.org/2021.tacl-1.49.pdf) [[repo]](https://github.com/pengbaolin/soloist)
- UBAR: Towards Fully End-to-End Task-Oriented Dialog System with GPT-2 [[paper]](https://arxiv.org/pdf/2012.03539.pdf) [[repo]](https://github.com/TonyNemo/UBAR-MultiWOZ)
- Improving End-to-End Task-Oriented Dialogue System with A Simple Auxiliary Task [[paper]](https://aclanthology.org/2021.findings-emnlp.112.pdf) [[repo]](https://github.com/bepoetree/MTTOD)

We thank the authors of the papers for releasing their high quality codes for the purpose of research development.

In addition, we would like to thank HuggingFace for all pre-trained models and libraries.
