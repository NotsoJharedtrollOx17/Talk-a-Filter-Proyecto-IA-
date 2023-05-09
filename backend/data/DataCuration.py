from transformers import pipeline
import time
import csv

SCRAPED_DATA = './data-files/csv/scraped-content.csv'
CURATED_DATA = './data-files/csv/curated-content.csv'
HUGGINGFACE_MODEL = "declare-lab/flan-alpaca-large"

data_curator = pipeline('text-generation' ,model=HUGGINGFACE_MODEL)
scraped_data_entries = []
curated_dataset = []

def load_offensive_words_dataset():
    with open(SCRAPED_DATA, 'r') as file:
        reader = csv.reader(file, delimiter='\n')
        for row in reader:
            scraped_data_entries.append(row[0])
        file.close()

    scraped_data_entries.remove(["Instruction", "Input"])

def generate_dataset_output():
    print("generate_dataset_output")
    print("\tBEGIN\t\n")
    for (i, item) in enumerate(scraped_data_entries, start=1):
        instruction = item[0]
        input = item[1]
        prompt = instruction + input
        output = data_curator(prompt, max_length=150, do_sample=True)
        curated_dataset.append([instruction, input, output])
        print(f"No. {i}:\n\tprompt: {instruction}\n\toutput: {output}\n")
        
        if i % 10 == 0:
            print(f"TIMEOUT {i}\n")
            time.sleep(2.5)

    print(F"\nSaving data to {CURATED_DATA}\n")
    with open(CURATED_DATA, mode='w', newline='') as data_file:
        writer = csv.writer(data_file)
        writer.writerow(["instruction", "input", "output"])
        writer.writerows(curated_dataset)
        data_file.close()
    print(f"\nDataset generated to {CURATED_DATA}\n")

    print("\tEND\t")

def main():
    print('Data curation for model')
    generate_dataset_output()

if __name__ == '__main__':
    main()