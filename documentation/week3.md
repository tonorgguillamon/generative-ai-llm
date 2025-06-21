# Reinforcement learning from human feedback (RLHF)

Define your model alignment criterion: why reward or penalize

Prompt -> Completions -> Rank -> Train reward model -> Use the reward model to fine-tune LLM with RL

## Reward hacking
Model cheats seeking for completions that score the higest but differs from the desired behaviour.
    How to avoid this? Pass the dataset to not only the RL-updated LLM model but to a Reference Model, which is frozen, and calculates KL Divergence Shift Penalty.
    One step beyond is to feedback-update the active model as PEFT, so it's lighter.

## Scaling human feedback
Helpful LLM -> Red Teaming -> Response Critique and Revision -> Fine-tuned LLM

# LLM-powered applications
## Model optimizations for deployment
- Distillation: use a LLM Teacher to "train" a LLM Student
Freeze LLM Teacher parameters, and pass training data to the LLM Student. The student makes predictions which injected into a loss function. The relationship between the labels from the LLM Teacher and the predictions from the LLM Student is the Distillation Loss.
Trick: increase temperature of the softmax function.
- Quantization: LLM -> 16-bit quantized LLM (reduce size of the model). This can be applied to model weights, and/or activation layers.
- Pruning: LLM -> remove redundant parameters -> Pruned LLM.
Remove model weights with values close or equal to zero.

## Using LLM in application
Models having difficulty:
- Out of date data
- Complex math
- Hallucinations: generate text even if it doesn't know the answer.

## Program-aided language models (PAL)
LLM + Code Interpreter
Transform input text into variables in the script.
Question -> PAL prompt template -> PAL formatted prompt -> LLM -> injection into Python script -> Python interpreter returns answer -> back to PAL formatted solution

## ReAct: Synergizing Reasoning and Action in LLMs
i.e. LLM + Websearch API
LangChain: Prompt Templates, Memory, Tools, Agents
LangGraph

## Architectures

