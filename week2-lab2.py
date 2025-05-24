"""
Fine-tune a Generative AI Model for Dialogue Summarization

FLAN-T5 model, which provides a high quality instruction tuned model and can summarize text out of the box.
Let's improve the interferences, and evaluate the results with ROUGE metrics. Then, we will perform PEFT, and compare the results.
"""
from datasets import load_dataset # datasets belongs to HuggingFace
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer
import torch
import time
import evaluate
import pandas as pd
import numpy as np

# Load dataset and LLM from HuggingFace
huggingface_dataset_name = "knkarthick/dialogsum" # contains 10,000+ dialogues with the corresponding manually labeled summaries and topics.
dataset = load_dataset(huggingface_dataset_name)

# Load pre-trained FLAN-T5 model, and its tokenizer, from HuggingFace.
# bffloat16 specifies the memory type, making it lightweight
model_name='google/flan-t5-base'

# AutoModel and AutoTokenizer recognize the model name and choose their specification accordingly
# AutoModelForSeq2Seq detects flan-t5, so internally loads T5ForConditionalGeneration 
# AutoTokenizer detects flan-t5, so interally uses T5Tokenizer
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# To extract the number of parameters we can iterate over each tensor and see its length:
all_model_params = 0
for _, param in original_model.named_parameters():
    all_model_params += param.numel()

    # with param.requires_grad (bool) you can also see whether the parameter will be updated during the training -> hence, trainable!

# Test model with zero-shot inferencing:
index = 200

dialogue = dataset['test'][index]['dialogue']
summary = dataset['test'][index]['summary']

prompt = f"""
Summarize the following conversation.

{dialogue}

Summary:
"""

inputs = tokenizer(prompt, return_tensors='pt') # pt -> pyTorch
output = tokenizer.decode(
    original_model.generate(
        inputs["input_ids"], 
        max_new_tokens=200,
    )[0], 
    skip_special_tokens=True
)

print(f'INPUT PROMPT:\n{prompt}\n')
print(f'BASELINE HUMAN SUMMARY:\n{summary}\n')
print(f'MODEL GENERATION - ZERO SHOT:\n{output}')

# MODEL GENERATION - ZERO SHOT:
#Person1#: I'm thinking of upgrading my computer. -> the model understands, although poorly...

##############################
## Perform Full Fine-Tuning ##
##############################

# 1. Preprocess the Dialog-Summary Dataset
"""
Convert the dialog-summary (prompt-response) pairs into explicit instructions for the LLM
Prepend an instruction to the start of the dialog and to the start of the summary.
i.e.:
    Training prompt (dialogue):
        Summarize the following conversation.

            Chris: This is his part of the conversation.
            Antje: This is her part of the conversation.
            
        Summary: 
    
    Training response (summary):
        Both Chris and Antje participated in the conversation.

Then, preprocess the pompt-response dataset into tokens and pull out their input_ids (1 per token)
These two parameters are addionally added to ones from the dataset, where each element has: "dialogue" and "summary"
"""

def tokenize_function(element):
    # element has -> dialogue, summary
    # we want to add more information to the dialogue -> reshape it with a layout
    start_prompt = 'Summarize the following conversation.\n\n'
    end_prompt = '\n\nSummary: '
    prompt = [start_prompt + dialogue + end_prompt for dialogue in element["dialogue"]]

    # the model ONLY understands vectors (numbers), so what we store are the ids, from both: the prompt and the summary
    element['input_ids'] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    element['labels'] = tokenizer(element["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids
    
    return element
 

# The dataset actually contains 3 diff splits: train, validation, test.
# The tokenize_function code is handling all data across all splits in batches.
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# to improve adding tokenizer as an argument for the function (avoid globals)
# tokenized_datasets = dataset.map(lambda e: tokenize_function(e, tokenizer), batched=True)

# remove what's not necessary for the model
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary',])

# subsample the dataset for this lab:
tokenized_datasets = tokenized_datasets.filter(lambda example, index: index % 100 == 0, with_indices=True)

# Ensure that all shapes match:
print(f"Shapes of the datasets:")
print(f"Training: {tokenized_datasets['train'].shape}")
print(f"Validation: {tokenized_datasets['validation'].shape}")
print(f"Test: {tokenized_datasets['test'].shape}")

print(tokenized_datasets)

# 2. Fine-tune with preprocessed dataset
