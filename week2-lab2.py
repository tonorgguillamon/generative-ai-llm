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

# The Hugging Face Trainer expects the training dataset to have certain standard field names:
# "input_ids": the tokenized input for the model.
# "labels": the expected output (target) tokens for supervised learning tasks like sequence-to-sequence (seq2seq).
 # Internally would be like: model(input_ids=batch["input_ids"], labels=batch["labels"])

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
# Doc for trainer
# https://huggingface.co/docs/transformers/main_classes/trainer

output_dir = f'./dialogue-summary-training-{str(int(time.time()))}'

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=1e-5,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_steps=1,
    max_steps=1
)

trainer = Trainer( # Hugging Face Trainer is used here to fine-tune (retrain) a pretrained model on your specific dataset
    model=original_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation']
)

trainer.train() # this might take few hours...

trainer.save_model("fine-tuned-model-lab2")

# to use the model:
model = AutoModelForSeq2SeqLM.from_pretrained("fine-tuned-model-lab2")
tokenizer = AutoTokenizer.from_pretrained("fine-tuned-model-lab2")

dialogue = "Chris: Hi, how are you?\nAntje: I'm good, thanks! How about you?"
prompt = f"Summarize the following conversation.\n\n{dialogue}\n\nSummary:"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(inputs["input_ids"], max_new_tokens=100)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(summary)

# 3. Evaluate the Model Qualitatively - ROUGE Metric
rouge = evaluate.load('rouge')

# For this we have to compare several summaries, and then evaluate the outcome.
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []

for _, dialogue in enumerate(dialogues):
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    original_model_outputs = original_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    original_model_text_output = tokenizer.decode(original_model_outputs[0], skip_special_tokens=True)
    # important: we take the first element of original_model_outputs[0] because the generate method of the model
    # returns a batch of generated sequences, even if you only provided one input!
    original_model_summaries.append(original_model_text_output)

    instruct_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries))

# to visualize the comparative better:
df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries'])


original_model_results = rouge.compute(
    predictions=original_model_summaries,
    references=human_baseline_summaries[0:len(original_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('ORIGINAL MODEL:')
print(original_model_results)
print('INSTRUCT MODEL:')
print(instruct_model_results)

# Conclusion:
# -----------------------------
# Absolute percentage improvement of INSTRUCT MODEL over HUMAN BASELINE
# rouge1: 18.82%
# rouge2: 10.43%
# rougeL: 13.70%
# rougeLsum: 13.69%

####################################################
## Perform Parameter Efficient Fine-Tuning (PEFT) ##
####################################################

"""
PEFT is a generic term that includes Low-Rank Adaptation (LoRA) and prompt tuning (which is NOT THE SAME as prompt engineering!).
In most cases, when someone says PEFT, they typically mean LoRA. LoRA, at a very high level, allows the user to fine-tune their model
using fewer compute resources (in some cases, a single GPU). After fine-tuning for a specific task, use case, or tenant with LoRA,
the result is that the original LLM remains unchanged and a newly-trained “LoRA adapter” emerges.
This LoRA adapter is much, much smaller than the original LLM - on the order of a single-digit % of the original LLM size (MBs vs GBs).

That said, at inference time, the LoRA adapter needs to be reunited and combined with its original LLM to serve the inference request.
The benefit, however, is that many LoRA adapters can re-use the original LLM which reduces overall memory requirements when serving multiple tasks and use cases.
"""

# 1. Setup the PEFT/LoRA model for Fine-tuning
# We need a new layer/parameter adapter. Using PEFT/LoRA we are freezing the underlying LLM and only training the adapter.
# The rank "r" hyper-parameter means: the dimension of the adapter to be trained.

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # rank -> higher means more trainable parameters (potentially more capacity) but also more memory usage
    lora_alpha=32, # scaling factor for the weights -> it balances the contribution of the adapter to the original weights (larger values, adapter's effect stronger)
    target_modules=["q", "v"], # which modules in the model to apply LoRA to. "q": querry, "v": value, projection layers in the attention mechanism of the transformer.
    lora_dropout=0.05, # to prevent overfitting
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # to let the model know what are we working with; FLAN-T5
)

# Now we are LoRA adapter to the original LLM to be trained:
peft_model = get_peft_model(original_model, 
                            lora_config)

# if we extract the number of parameters of the model, and trainable ones:
"""
trainable model parameters: 3538944
all model parameters: 251116800
percentage of trainable model parameters: 1.41%

This is because ONLY the adapter is trainable. The original model weights are frozen.
"""
# 3. Train PEFT adapter
output_dir = f'./peft-dialogue-summary-training-{str(int(time.time()))}'

peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3, # Higher learning rate than full fine-tuning.
    num_train_epochs=1,
    logging_steps=1,
    max_steps=1    
)
    
peft_trainer = Trainer(
    model=peft_model,
    args=peft_training_args,
    train_dataset=tokenized_datasets["train"],
)

peft_trainer.train() # we are training LoRA model with our dataset. It's way more efficient cause we update ~1% of the parameters

# after the training, the adapter has "learned" how to adapt the base model to our specific summarization task

peft_model_path="./peft-dialogue-summary-checkpoint-local"

peft_trainer.model.save_pretrained(peft_model_path)
tokenizer.save_pretrained(peft_model_path)

# For using the model:
# --------------------------
from peft import PeftModel, PeftConfig

model_base = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

peft_model = PeftModel.from_pretrained(model_base, 
                                       './peft-dialogue-summary-checkpoint-from-s3/', 
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False) # since we are now only gonna use the model, we don't need to carry parameters for training

# 3. Evaluate the model quantitatively with ROUGE metric
dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

original_model_summaries = []
instruct_model_summaries = []
peft_model_summaries = []

for idx, dialogue in enumerate(dialogues):
    prompt = f"""
Summarize the following conversation.

{dialogue}

Summary: """
    
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    human_baseline_text_output = human_baseline_summaries[idx]

    peft_model_outputs = peft_model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    peft_model_text_output = tokenizer.decode(peft_model_outputs[0], skip_special_tokens=True)

    peft_model_summaries.append(peft_model_text_output)

zipped_summaries = list(zip(human_baseline_summaries, original_model_summaries, instruct_model_summaries, peft_model_summaries))
 
df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'original_model_summaries', 'instruct_model_summaries', 'peft_model_summaries'])

peft_model_results = rouge.compute(
    predictions=peft_model_summaries,
    references=human_baseline_summaries[0:len(peft_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

"""
3 models evaluated:
ORIGINAL MODEL:
{'rouge1': 0.2127769756385947, 'rouge2': 0.07849999999999999, 'rougeL': 0.1803101433337705, 'rougeLsum': 0.1872151390166362}
INSTRUCT MODEL:
{'rouge1': 0.41026607717457186, 'rouge2': 0.17840645241958838, 'rougeL': 0.2977022096267017, 'rougeLsum': 0.2987374187518165}
PEFT MODEL:
{'rouge1': 0.3725351062275605, 'rouge2': 0.12138811933618107, 'rougeL': 0.27620639623170606, 'rougeLsum': 0.2758134870822362}

Notice, that PEFT model results are not too bad, while the training process was much easier!!
"""
