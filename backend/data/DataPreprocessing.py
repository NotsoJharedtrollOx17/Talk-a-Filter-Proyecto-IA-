from datasets import load_dataset
from transformers import AutoTokenizer

CURATED_DATA = './data-files/csv/curated-content.csv'
HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-large"
TOKENIZED_DATASET = './data-files/processed-dataset/tokenized-curated-content'

# Load the dataset from the CSV file
dataset = load_dataset('csv', data_files=CURATED_DATA)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

# Tokenize the text data
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def main():
    dataset = dataset.map(tokenize_function, batched=True)

    # Print the first example in the dataset
    print(dataset[0])

    # Save the processed dataset to disk
    dataset.save_to_disk(TOKENIZED_DATASET)

if __name__ == '__main__':
    main()
