import markovify

# Training text corpus
text_corpus = """
Generative AI refers to deep-learning models that can create high-quality text, images, and other content based on training data.
Artificial intelligence has advanced rapidly over recent years, transforming how we interact with technology.
Machine learning empowers algorithms to learn from patterns in data without being explicitly programmed.
Generative AI models learn the underlying statistical patterns of data to produce original synthetic outputs.
Deep learning models use multi-layer neural networks to analyze complex data structures effectively.
Statistical models like Markov chains predict the probability of a word based on preceding sequence history.
"""

# Build Markov chain model (state_size=2 considers pairs of words)
text_model = markovify.Text(text_corpus, state_size=2)

# Generate synthetic sentences
print("--- Generated Sentences using Markov Chains ---")
for i in range(3):
    sentence = text_model.make_sentence()
    if sentence:
        print(f"{i+1}. {sentence}")
    else:
        print(f"{i+1}. {text_model.make_short_sentence(100)}")
