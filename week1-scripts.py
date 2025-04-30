# LAB 1
# Generatve AI Use Case: Summarize Dialogue
# Amazon SageMaker

from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

huggingface_dataset_name = "knkarthick/dialogsum" # 10000 dialogues with the corresponding manually labeled summaries and topics

dataset = load_dataset(huggingface_dataset_name)

#print(dataset['test'][0]['dialogue']) # 0 is just the index within the datasrt
#print(dataset['test'][0]['summary']) # summary is what the user has summarizedlabelled from this dialogue

model_name = "google/flan-t5-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True) # raw text to vectored space

# Example to tes the tokenizer encoding and decoding:
sentence = "What time is it, Luka?"
sentence_encoded = tokenizer(sentence, return_tensors='pt')
# sentence enconded outputs a dictionar containing various keys: input_ids, attention_mask, etc.
# input_ids: contains the tokenized representation of the input sentence. It's a tensor (PyTorch format)
# where each token in the sentence is mapped to its corresponding numerical ID from the tokenizer's vocabulary

# the tokenizer outputs a batch, even for a single sentence. We need to access the first element to get the token IDs - [0].
sentence_decoded = tokenizer.decode(sentence_encoded['input_ids'][0],
                                    skip_special_tokens=True)

print(f"Encoded sentence: {sentence_encoded['input_ids'][0]}")
print(f"Decoded sentence: {sentence_decoded}")

# Get what the model is generating as a summary for the conversations:
for index in range(5):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']

    inputs = tokenizer(dialogue, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs['input_ids'],
            max_new_tokens=50
        )[0],
        skip_special_tokens=True
    )

    print(f"Example {index}")
    print(f"Input prompt: {dialogue} \n")
    print(f"Human summary: {summary} \n")
    print(f"Model generation - without prompt engineering: {output} \n")
    print(f"-----------------------------------")

# Add prompt engineering to help the model!
# Zero shot
for index in range(5):
    dialogue = dataset['test'][index]['dialogue']
    summary = dataset['test'][index]['summary']
    
    # Option 1:
    prompt1 = f"""
    Summarize the following conversation {dialogue}
    Summary:
    """
    # Option 2:
    prompt2 = f"""
    Dialogue {dialogue}
    What was going on?
    """

    inputs = tokenizer(prompt2, return_tensors='pt')
    output = tokenizer.decode(
        model.generate(
            inputs['input_ids'],
            max_new_tokens=50
        )[0],
        skip_special_tokens=True
    )

    print(f"Example {index}")
    print(f"Input prompt: {dialogue} \n")
    print(f"Human summary: {summary} \n")
    print(f"Model generation - without prompt engineering: {output} \n")
    print(f"-----------------------------------")

# Let's improve it and use one shot prompt and few shots prompts
def make_prompt(indexes, indexToSummarize):
    prompt = ''
    for i in indexes:
        dialogue = dataset['test'][i]['dialogue']
        summary = dataset['test'][i]['summary']

        prompt += f"""
        Dialogue:
        {dialogue}
        What was going on?
        {summary}
        """
    
    dialogue = dataset['test'][indexToSummarize]['dialogue'] # we are giving context!
    # accumulating the other conversation and then eventually asking to summarize one of them
    prompt += f"""
    Dialogue:
    {dialogue}
    What was going on?
    """
    return prompt

oneShotPrompt = make_prompt(indexes=[3], indexToSummarize=20) # indexes is just one! this is for ONE shot prompt
fewShotPrompt = make_prompt(indexes=range(5), indexToSummarize=20)
