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

from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
import numpy
import random

text_to_text_tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
text_to_text_model = T5ForConditionalGeneration.from_pretrained(
    "gooogle/fraln-t5",
    devicemap = "auto"
)

text = "Fucking loss porn in that post man!"

input_text = r"Rewrite the following text for it to not contain foul language: {text}"
input_ids = text_to_text_tokenizer(input_text, return_tensors="pt", padding=True).input_ids.to("GPU_OF_COLAB_SERVER")
outputs = text_to_text_model.generate(input_ids)
rewritten_text = text_to_text_tokenizer.decode(outputs[0])

print(rewritten_text)

def main():
    training_arguments = TrainingArguments(
        output_dit="wallstreetbets-rewriter",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        save_strategy="epoch",
    )

    # TODO add training & evaluation datasets for the sake of the model
    trainer = Trainer(
        model = text_to_text_model,
        args = training_arguments,
        train_dataset = ,
        eval_dataset = ,
    )

    trainer.train()

    trainer.evaluate()

if __name__ == "__main__":
    main()