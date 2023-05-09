''''
            LEER CADA UNO PARA ENTRENAR MODELO HIPOTETICO
* https://huggingface.co/docs/transformers/main_classes/pipelines
* https://towardsdatascience.com/teach-an-ai-model-to-write-like-shakespeare-for-free-a9e6a307139
* https://neptune.ai/blog/how-to-use-google-colab-for-deep-learning-complete-tutorial
* https://huggingface.co/blog/sentiment-analysis-python
* https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
* https://www.geeksforgeeks.org/scraping-reddit-using-python/
* https://huggingface.co/docs/transformers/generation_strategies
* https://huggingface.co/blog/sentiment-analysis-python
* https://huggingface.co/docs/transformers/training
* https://huggingface.co/google/flan-t5-base
* https://wandb.ai/authors/Kaggle-NLP/reports/Kaggle-s-NLP-Text-Classification--VmlldzoxOTcwNTc
* https://jmlr.org/papers/volume21/20-074/20-074.pdf Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
* https://arxiv.org/pdf/2210.11416.pdf Scaling Instruction-Finetuned Langueage Models
* https://github.com/JosephLai241/URS Universal Reddit Scraper
'''

# TODO ejecutar codigo para utilizar en Google Colab para entrenar el modelo
'''
La idea principal es detectar si un texto es ofensivo y 
    de ser verdadero, reescribir el texto para que deje de serlo;
    De lo contrario el texto se mantiene igual para no hacer el proceso de okis...
'''

from datasets import load_from_disk, train_test_split, DatasetDict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, DataCollatorWithPadding, Trainer, TrainingArguments
import torch

HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-large"
TOKENIZED_DATASET = './data-files/processed-dataset/tokenized-curated-content'

def load_dataset():
    dataset = load_from_disk(TOKENIZED_DATASET)
    train_dataset, eval_dataset = train_test_split(dataset, test_size=0.2, shuffle=True)
    return DatasetDict({ 'train': train_dataset, 'eval': eval_dataset })

def compute_metrics(eval_pred):
    predictions, label_ids = eval_pred
    preds = np.argmax(predictions, axis=1)
    return {"accuracy": (preds == label_ids).mean()}

def set_training_arguments():
    arguments = TrainingArguments(
        output_dir='./ai-model',          # output directory
        evaluation_strategy='epoch',    # evaluation strategy to adopt during training
        num_train_epochs=3,             # total number of training epochs
        per_device_train_batch_size=16, # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        learning_rate=2e-5,             # learning rate
        weight_decay=0.01,              # strength of weight decay
        push_to_hub=False,              # whether to push model checkpoint to the Hub
        logging_dir='./ai-model/logs',           # directory for storing logs
        logging_steps=10,
        load_best_model_at_end=True,    # load the best model when finished training
        metric_for_best_model='accuracy',
    )
    
    return arguments

def set_model():
    return AutoModelForSeq2SeqLM.from_pretrained(HUGGINGFACE_MODEL)

def training():
    dataset_dictionary = ''
    training_args = ''
    model_to_train = ''
    data_collator = ''
    tokenizer = ''

    print("\tBEGIN\t\n")

    set_training_arguments()
    tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
    dataset_dictionary = load_dataset()
    model_to_train = set_model()
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model_to_train,
        args=training_args,
        train_dataset=dataset_dictionary['train'],
        eval_dataset=dataset_dictionary['eval'],
        compute_metrics=compute_metrics(),
        data_collator=data_collator
    )

    trainer.train()

    print("\tEND\t\n")

def main():
    training()

if __name__ == '__main__':
    main()