from datasets import load_dataset
from transformers import AutoTokenizer

CURATED_DATA = './curated-content.csv'
TOKENIZED_DATASET = './tokenized-curated-content'
HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-large"

# Load the dataset from the CSV file
dataset = load_dataset('csv', data_files=CURATED_DATA)

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)

# Tokenize the text data
# TODO change tokenizer and see if this can be implemented to the training process
def tokenize_function(examples):
    tokens = tokenizer(
        examples['text'], 
        padding='max_length', 
        truncation=True,
        return_tensors='pt'
      )
    
    return tokens
    
def main():
    print("Saving processed dataset")
    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Print the first example in the dataset
    print(tokenized_dataset['train'][0])

    # Save the processed dataset to disk
    tokenized_dataset.save_to_disk(TOKENIZED_DATASET)
    print("Preprocessed dataset saved!")

if __name__ == '__main__':
    main()