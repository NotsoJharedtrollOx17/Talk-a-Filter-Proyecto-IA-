from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, EncoderDecoderModel, DataCollatorWithPadding, Trainer, TrainingArguments
from torch.utils.data import random_split
import numpy as np
import pandas as pd
import torch
import evaluate

HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-base"
TOKENIZED_DATASET = './tokenized-curated-content/train'

# Type of metric
metric = evaluate.load("accuracy")

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    preds = np.argmax(predictions, axis=1)
    return metric.compute

def set_training_arguments():
    arguments = TrainingArguments(
        output_dir='./ai-model',          # output directory
        evaluation_strategy='epoch',    # evaluation strategy to adopt during training
        num_train_epochs=3,             # total number of training epochs
        per_device_train_batch_size=2, # batch size per device during training
        per_device_eval_batch_size=3,  # batch size for evaluation
        learning_rate=2e-5,             # learning rate
        weight_decay=0.01,              # strength of weight decay
        push_to_hub=False,              # whether to push model checkpoint to the Hub
        logging_dir='./ai-model/logs',           # directory for storing logs
        logging_steps=10,
        metric_for_best_model='accuracy',
    )
    
    return arguments

def set_model():
    return EncoderDecoderModel.from_encoder_decoder_pretrained(HUGGINGFACE_MODEL, HUGGINGFACE_MODEL)

def tokenize_function(examples, id):
  if id == 'text':
    tokens = tokenizer(
        examples['input'], 
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
      ).input_ids
  else:
      tokens = tokenizer(
        examples['output'], 
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
      ).input_ids

  return tokens

def training():
    model_to_train = ''

    print("\tBEGIN\t\n")

    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
    # Load the dataset from the CSV file
    dataset = load_dataset('csv', data_files=CURATED_DATA)
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    text = tokenize_function(dataset, 'text')
    output = tokenize_function(dataset, 'not text')
    model_to_train = set_model()
    model_to_train.config.decoder_start_token_id = tokenizer.cls_token_id
    model_to_train.pad_token_id = tokenizer.pad_token_id

    loss = model(input_ids=text, labels=output).loss

    print(loss)

    print("\tEND\t\n")

def main():
    training()

if __name__ == '__main__':
    main()