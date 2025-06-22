# AI/ML Text Processing Notes

## Introduction

This is the beginning of the notes.

## Goals

- Understand the challenges of converting unstructured text into numerical data for AI/ML
- Store vectors effectively

## Agenda

1. Turn text to numbers
2. Improving the representations
3. Storing Embeddings

---

## Part 1: Turning Text into Numbers

### Why Convert Text to Numbers?

Text-to-number conversion enables:

- Feeding data to ML/DL models
- Information retrieval
- Document summarization
- Language translation
- Content recommendation
- Sentiment analysis
- Question answering systems

### Data Representation Objectives

- Obtain a unified semantic space
- Create a single representation that works with all types of words
- Ensure practical usability of the representations

![Text to Numbers Visualization](photo1.png)
_Figure 1: Process of converting text to numerical representations_

---

### Challenges in Text Representation

1. **Dimensionality**: High-dimensional sparse vectors
2. **Semantics**: Capturing word meanings and relationships
3. **Context**: Handling polysemy (multiple meanings)
4. **Scalability**: Processing large text corpora

## Part 2: Tokenization and first models

### Tokenization

Tokens are not always words. There is not a fixed way to create tokens. It is an art to create tokens.

### Problems:

- contractions
- some words that are very repitive (like 'the', 'a')
- if pictogram languages like Chinese.

![Tokenization](photo2.png)
_Figure 2: Tokenization_

### Approaches

1.  **_Rule-Based Systems_** : say for a spam detection were based on rules such as repetition of words, lots of exclamation points or some patterns in spam messages. Then create a vector of those rules. Or count the frequency of appearance of those rules and create a vector. Problem:

- for new types of spam messages, need to keep on updating rules.
- Rules can also be biased

2.  **_Bag of Words_** : The first statistical model. Two documents each with single sentence. Create a vocabulary, don't keep repeated ones. Now make two bags: first with tokens of one sentence and second with tokens from another sentence to check. Now, create a vector of the words that are matching with the tokens in the vocabulary for both individually and combine these vectors to form a matrix. Columns represent documents, rows represent words. Word representation in various documents is our row.
    ![Tokenization](photo3.png)
    _Figure 3: Bag of Words_

### Advantages of Bag of Words:

- Simple, efficient, language agnostic, interpretability, useful for certain tasks

### Disadvantages

- lost information about the sequence(don't take into account how words are sorted, I love NLP, and I NLP love are the same for this )
- Fixed vocab size, equal importance to all, inefficient with large dataset, can't handle out of vocab words(when required a synonym?)

3. **_ tf-idf _** : Term frequency, inverse document frequency. Counting not only if the word is present or not but the frequency of the presence of the words.
   The problem? Some words might be repeated too often. Solution: take into consideration the frequency of occurrence of a word in that document and in all documents (see if a word appears in that particular document and then in other doucments or not) and compare term frequency to document frequency. Create vectors of them. Then assign weights to each term.

![Tokenization](photo4.png)
_Figure 4: tf-idf_

### Advantages of tf-idf:

- paying attention to content relevance
- Language agnostic
- weighted representation

### Disadvantages of tf-idf:

- sparse vectors (suppose in a million dimensional vector only 10 of the elements are 1 while all others are 0 (most are 0)? still computationally expensive)
- manual tuning required
- can't handle misspellings, ignores semantic meaning

---

## Key Concepts

### Text Vectorization Methods

| Method       | Description                          | Use Case                  |
| ------------ | ------------------------------------ | ------------------------- |
| Bag-of-Words | Count-based word representation      | Basic text classification |
| TF-IDF       | Frequency-inverse document frequency | Information retrieval     |
| Word2Vec     | Neural word embeddings               | Semantic analysis         |
| BERT         | Contextual embeddings                | Advanced NLP tasks        |

## Improving the Representations

### Lexical Semantics

- Previous algorithms don't take into account the semantics and connection of words.
- want to work with multiple meanings(polysemy)(the word bank has so many meanings)
- synonyms
- word similarity(dogs and cats are similar in the animal aspect)
- Word relatedness(are not similar but somehow related like cup and coffee)
- Semantic frames and roles: Depends on the context of the sentence. (sub, verb, object)(buyer and seller)
- Connotations: how the text has been written( sarcasm, irony, literally, ...)

## The concept of Embedding

- Vectors that come to fill the above needs
- They are not sparse anymore, they are usually trained on tons of documents.
- Deep Learning Algorithm for training the embeddings.
- It is technically any kind of text representation

### Analogy Questions:

- Simplified to 2D space
- first the vector for king, subtract that vector to man and add the vector to woman.
- it is closer to queen

![Analogy Questions](photo5.png)
_Figure 5: Analogy-Questions_

### Historical Semantics

- How the meaning of the word has evolved over time?

### A small thing about the learned semantics:

- the learned semantics don't necessarily correspond to the interpretation that we give to those words.

- those semantics are learnt from millions of texts, mostly from the internet

- they represent the average meaning of the texts that we have used for creating those representations.

- the bias and prejudices present in the texts are also contained in our representations.

## Interlude: Metrics and Visualization

### Vector Comparison:

![Vector Comparison](photo6.png)
_Figure 6: Vector_Comparison_

- Cosine similarity best for recommendation systems

### Vector Visualization

- we live in space that is 3D, but we have a vector of millions of dimensions.
- Need to reduce the dimensions of vectors:

1. PCA(Principal Component Analysis): only work with linear basis(limitations)
2. t-SNE (t-distributed stochastic neighbor embedding): it does not have linearity limitation. But it is slow.
3. UMAP(Uniform Manifold Approximation and Projection) : Really convenient for very high dimensional vectors. It is one of the standards for visualization of vectors nowadays.

- Example in MNIST dataset:
  ![Vector Visualization](photo7.png)
  _Figure 7: Vector_Visulization_

- Note that PCA is having trouble visualizing all the classes properly while others are doing it properly.

### Limitations of these techniques

- Information loss (reducing dimensions loses information)
- Overcrowding and clutter ( if too many vectors)
- Interpretation Challenges(even though 2D or 3D but cannot interpret it easily)

- Example how a 3D elephant is converted to 2D becomes hard to interpret.

![Example](photo8.png)
_Figure 8: Example_of_limitation_

## Improving the Representations (continued)

### word2vec

- We have this in two flavors:

1. Continuous Bag-of-Words(CBOW):
   We give inputs to our model a window of token around a central word and model tries to predict what is between the tokens.
   We have : The dominant, transduction models, and sequence.
   ![Continuous Bag of Words](photo9.png)
   _Figure 9: Continuous_Bag_of_Words_Visualization_

2. Skip-Gram: Opposite of above.
   ![Skip Gram](photo10.png)
   _Figure 10: Skip-Gram_

- Can still improve the prediction power by some techniques:
- Negative sampling: choose those tokens that we know for sure are neighbors
- Sub Sampling: for the words that appear really often, no need to check always

- those models are based on fake task, only used for creating the word embeddings

- it is old, but now improved in:

### GloVe

- Balances Global and Local Context

### FastText

- Handles out-of-vocabulary words

### Doc2Vec

- Unsupervised Learning of Document Embeddings

![Other Similar Embeddings](photo11.png)
_Figure 11: Other_Similar_Embeddings_

#### All of these models are good but they generate what is called as static embeddings

- They have fixed vectors for all words, but what if for words that have multiple meanings, need contextual embeddings.

- Use Contextual embeddings: they are flexible, we need word and context and it would generate different vectors for same word.

![Static vs Contextual Embeddings](photo12.png)
_Figure 12: Static_Vs_Contextual_Embeddings_

- But how to do that?
- Development of more complex embeddings:
- CNN -> RNNs-> LSTMs->GRUs->Attention Mechanics -> Transformers

### Sentence Embeddings and Sentence Transformers

- What can we do if we are looking for context, take the average of all the embeddings, can be a time consuming

- SBERT has been developed for it.

### Getting Better Embeddings

- Knowledge Graphs
- Multimodality

## Storing Embeddings

- Need special databases: Vector Databases
- Databases can store vector embeddings as if it was a relational database

### Vector Indices and Vector Databases

- Indexing of vectors: gets an embedding and makes an index of it and stores in the database, and we can query it.
- Index, query, and post-processing
- Data Management, Metadata storage and filtering, Backups, Security, Integration,

- Examples of Vector Database:

![Examples of Vector Databases](photo13.png)
_Figure 13: Examples_of_Vector_Databases_

## Summing it Up:

![Summary](photo14.png)
_Figure 14: Summary_

## Other Sources for Further Info: 

![Further Sources](photo15.png)
_Figure 15: Other_Sources_


