https://www.coursera.org/learn/generative-ai-with-llms/home


Transformers:
- RNN vs LLMs

- How does it work?

Sentences -> tokenization (same as for training the model) -> Embedding (adding the position of each word in the sentence) -> Encoder (contextual understanding and produces one vector per input token - which stands for the meaning of the word within the sentence) -> Decorder (accepts input tokens and calculate the probability of each word from the WHOLE vocabulary to be next) -> Softmax output (select the most likelikely word). In every iteration the output is fed into the Embedding of the Decoder, to give more information to the model.
Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting

- Encoder Only Models: sequence2sequence - input and output sequences are of the same length. i.e.: classifications tasks (sentiment analysis)
- Encoder Decoder Models: i.e.: translations, generation of text.
- Decorder Only Models: you can generalize these models to most applications. i.e.: GPT, Llama

Prompt Engineering:
- Help the model to understand the context. Provide examples of the task you want the model to carry out: in-context learning.
- Warning! Context Window is limited in words. Use finetuning if not able to improve the response of the model.

Generative Configuration
- Inference parameters -> to control the interaction:
    * max. number of tokens
    * greedy vs random sampling (taking the word/token with the highest probability vs random-weighted sampling which gives the model a greater creativity to pick the next word)
        - "top k": select from within the top k results after applying random-weighted strategy using the probabilities.
        - "top p": select from the top ranked consecutive results by probability and with a cumulative probability <= p. If "top p" is 0.3, and the 3 first words have 0.2, 0.1 and 0.05 probabilities, it would only choose from within the first and the second word, since 0.2 + 0.1 are already reaching the parameter "p".
    * temperature, influences the shape of the probability distribution that the model calculates for the next token. The higher the temperature the higher the randomness and viceverse. Low temperature (<1), the probability distribution is highly peaked over the highest probability, giving barely no chance to the others; whereas a high temperature (>1), the probability distribution is flatter.
