from datasets import load_dataset, Dataset
from transformers import T5Tokenizer

DATASET = "my_data.csv"
TOKEN_MODEL = "google/flan-t5-base"

# Load the dataset
dataset = load_dataset("csv", data_files=DATASET)

# Load the tokenizer
tokenizer = T5Tokenizer.from_pretrained(TOKEN_MODEL)

# Define the preprocessing function
def preprocess_data(example):
    # Tokenize the text and map the labels to integers
    tokenized = tokenizer(example["text"], truncation=True, padding="max_length")
    label_map = {"OFFENSIVE": 1, "NON OFFENSIVE": 0}
    label_id = label_map[example["label"]]
    # Return the tokenized input and label ID
    return {"input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"], "labels": label_id}

# Apply the preprocessing function to the dataset
preprocessed_dataset = dataset.map(preprocess_data, batched=True)

# saving the preprocessed dataset to disk
dataset_to_save = Dataset.from_dict(preprocessed_dataset)
dataset_to_save.save_to_disk("wallstreetbet_posts_comments")