from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import pandas
import time
import csv

SCRAPED_DATA = './scraped-content.csv'
CURATED_DATA = './curated-content.csv'
HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-large"

tokenizer = AutoTokenizer.from_pretrained(HUGGINGFACE_MODEL)
data_curator = pipeline('text2text-generation', model=HUGGINGFACE_MODEL, tokenizer=tokenizer)
scraped_data_entries = []
curated_dataset = []


def load_offensive_words_dataset():
    scraped_data_df = pd.read_csv(SCRAPED_DATA)
    for index, row in scraped_data_df.iterrows():
        instruction = row['instruction'].strip()
        input = row['input'].strip()
        scraped_data_entries.append((instruction, input))

def generate_text(prompt):
    output = data_curator(prompt, max_length=128, do_sample=True)

    return output

def generate_dataset_output():
    load_offensive_words_dataset()

    print("generate_dataset_output")
    print("\tBEGIN\t\n")
    for (i, item) in enumerate(scraped_data_entries, start=1):
        instruction, input = item 
        prompt = instruction + input
        result = generate_text(prompt)
        output = result[0] 
        text = f"Below is an instruction that describes a task. Write a response that appropriately completes the request. ### Instruction: {instruction} ### Input: {input} ### Response: {output['generated_text']}"

        curated_dataset.append([instruction, input, output['generated_text'], text])
        print(f"No. {i}:\n\tprompt: {prompt}\n\toutput: {output['generated_text']}\n\ttext: {text}")
        
        if i % 10 == 0:
            print(f"TIMEOUT {i}\n")
            time.sleep(0.5)

    print(F"\nSaving data to {CURATED_DATA}\n")
    with open(CURATED_DATA, mode='w', newline='') as data_file:
        writer = csv.writer(data_file, lineterminator='\n')
        writer.writerow(["instruction", "input", "output", "text"])
        writer.writerows(curated_dataset)
        data_file.close()
    print(f"\nDataset generated to {CURATED_DATA}\n")

    print("\tEND\t")

def main():
    print('Data curation for model')
    generate_dataset_output()

if __name__ == '__main__':
    main()