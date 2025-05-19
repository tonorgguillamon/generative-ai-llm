# Fine-tuning LLMs with instruction
Before we saw how to improve the output of a model by in-context learning. However, this has limitations:
    - May not work for smaller models.
    - Examples take up space in the context window

Thus, fine-tuning is a great alternative!

* Using prompts to fine-tune LLMs with instruction:
    Add an "intruction"/request of what to do with the input. i.e.: "classify this review: I love this movie"
    multiple examples of: prompt[...], completion[...]

    pre-trained LLM -> trask-spefici examples -> fine-tuned LLM

    More examples:
        Summarize the following text:
            [EXAMPLE TEXT]
            [EXAMPLE COMPLETION]
        
        Translate this sentence to:
            [EXAMPLE TEXT]
            [EXAMPLE COMPLETION]

    Nonetheless, you must be carefull with memory usage!

* Fine-tuning on a single-task -> we could incurre into catastrophic forgetting!
    To avoid this:
        - fine-tune on multiple tasks at the same time
            requires many examples of each need for training
                Use FLAN family of models (Fine-tuned LAnguage Net): refer to a specific set of instructions used to perform instruction for fine-tuning (i.e. FLAN-T5, FLAN-PALM).

        - consider Parameter Efficient Fine-tuning (PEFT)

* Model evaluation: metrics
    - ROUGE: for text summarization, compares a summary to one or more reference summaries.
        > ROUGE-1 Recall = unigram matches / unigrams in reference
        > ROUGE-1 Precision = unigram matches / unigrams in output
        > ROUGE-1 F1 = 2 x (precision x recall) / (precision + recall)

        HOWEVER, these methods don't cover order of words. The model might score well, and the summary be a disaster.
        Hence, a better method takes into a consideration pair of words (bigrams).
        > ROUGE-2 Recall = bigram matches / bigrams in reference
        > ROUGE-2 Precision = bigram matches / bigrams in output
        > ROUGE-2 F1 = 2 x (precision x recall) / (precision + recall)

        Still, this might NOT tell much about the actual quality of the model.
        ROUGE-L is an alternative which takes the longest common subsequence (LCS)
        i.e.:
            reference        -> It is cold outside (4 unigrams)
            generated output -> It is very cold outside (5 unigrams)

            In this case LCS is "It is" and "cold outside", each with a length of 2.

        > ROUGE-L Recall = LCS(Gen, Ref) / unigrams in reference --> in the example: 2 / 4 = 0.5
        > ROUGE-L Precision = LCS(Gen, Ref) / unigrams in output --> 2 / 5 = 0.4
        > ROUGE-L F1 = 2 x (precision x recall) / (precision + recall) --> 2 x (0.2 / 0.9) = 0.44

        Nonetheless, these metrics can be fooled in some cases such as: the same word repeated multiple times (it will score a lot for unigram matches, since the word itself is well generated), or the many words from the reference guessed but shuffled - it's matching the unigrams, however if it's not in order the generation is poor!

    - BLEU SCORE: for text translation, compares to human-generated translations. BiLingual Evaluation Understanding.
        BLEU metric is an average precision across range of n-gram sizes.

    
    - Benchmarks for massive models:
        . Massive Multitask Language Understanding (MMLU)
        . BIG-bench Hard
        . Holistic Evalution of Language Models (HELM)

# Parameter efficient fine-tuning (PEFT)


    