from datasets import load_dataset
from transformers import AutoTokenizer

CURATED_DATA = './data-files/csv/curated-content.csv'
TOKENIZED_DATASET = './data-files/processed-dataset/tokenized-curated-content'
HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-large"

# Load the dataset from the CSV file
dataset = load_dataset('csv', data_files=CURATED_DATA)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

# Tokenize the text data
# TODO change tokenizer and see if this can be implemented to the training process
def tokenize_function(examples):
    return tokenizer(
        examples['instruction'],
        examples['input'],
        examples['output']  
        padding='max_length', 
        truncation=True, 
        max_lentgh=128
        )

def main():
    dataset = dataset.map(tokenize_function, batched=True)

    # Print the first example in the dataset
    print(dataset[0])

    # Save the processed dataset to disk
    dataset.save_to_disk(TOKENIZED_DATASET)

if __name__ == '__main__':
    main()
