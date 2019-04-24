# Generating "Genius" Annotations by Translating Lyrics with Seq2Seq/LSTM/RNN
# Neural Machine Translation: Meaning Behind Lyrics
![](images/genius_header2.png)
### Using Seq2Seq to Generate Lyric Annotations for Genius.com

*Capstone III Project for Galvanize Data Science Immersive, Week 12*

*by Taite Sandefer*

*Last Updated: 4/8/19*

## Table of Contents
- [Introduction](#introduction)
  - [Tech Used](#tech-used)
  - [Background](#background)
  - [Hypothesis and Assumptions](#hypothesis-and-assumptions)
  - [Methodology](#methodology)
- [Data Overview](#data-overview)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Engineered Features](#engineered-features)
  - [Challenges](#challenges)
- [Model Selection](#model-selection)
  - [Text Preprocessing](#text-preprocessing)
  - [Model Architecture](#model-architecture)
  - [Training Corpus](#training-corpus)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Performance Metrics](#performance-metrics)
- [Chosen Model](#chosen-model)
  - [Specifications](#specifications)
  - [Model Assessment](#model-assessment)
  - [Results and Interpretation](#results-and-interpretation)
- [Discussion](#discussion)
- [Acknowledgements](#acknowledgements)
  - [Citations](#citations)


# Introduction
## Background
### What is Genius?
[Genius](https://genius.com/), formerly "Rap Genius," is a website where users can view and add annotations to lyrics that help explain their meaning and context.

<p align="center">
  <img src="images/genius_biggest_collection.png" width = 400>
</p>

The primary goal of Genius is to explain lyrics and help make them more accessible to listeners. Generally, these are explanations regarding the semantic and/or cultural meanings behind lyrics, which can often cryptic and filled with linguistic subtleties that we wouldn't normally expect a computer to be able to pick up on.

<p>
  <img src="images/how_genius_works.png" width = 370 height = 145>  
  <img src="images/genius_annotation.png" width = 430 height = 150>
  <img src="images/genius_demo2.gif" width = 800>
</p>


* note to self: great explanation of Factorization, should try to mimick: https://github.com/declausen/capstone_project

* Graphing:
  -


* Demo
  - set of unseen lyrics for people to pick from (extensive list, and short list)

**try to mimick the tensorflow tutorial!! - how to build your neural machine translator**

# Seq2Seq / LSTMs

# Neural Machine Translation (NMT)

## Encoder/Decoder Architecture

- [thought vector](https://www.theguardian.com/science/2015/may/21/google-a-step-closer-to-developing-machines-with-human-like-intelligence)

> "Specifically, an NMT system first reads the source sentence using an encoder to build a "thought" vector, a sequence of numbers that represents the sentence meaning; a decoder, then, processes the sentence vector to emit a translation, as illustrated in Figure 1. This is often referred to as the encoder-decoder architecture. In this manner, NMT addresses the local translation problem in the traditional phrase-based approach: it can capture long-range dependencies in languages, e.g., gender agreements; syntax structures; etc., and produce much more fluent translations as demonstrated by Google Neural Machine Translation systems."
#### In Training, feed the model:
- encoder_inputs [max_encoder_time, batch_size]: source input words.
- decoder_inputs [max_decoder_time, batch_size]: target input words.
- decoder_outputs [max_decoder_time, batch_size]: target output words, these are decoder_inputs shifted to the left by one time step with an end-of-sentence tag appended on the right.

### Embedding
### Encoder
### Decoder
### Loss
### Gradient Computation & Optimization

### Secret Sauce: Attention Mechanism
> "The key idea of the attention mechanism is to establish direct short-cut connections between the target and the source by paying "attention" to relevant source content as we translate. A nice byproduct of the attention mechanism is an easy-to-visualize alignment matrix between the source and target sentences (as shown in Figure 4)."

- [Bahdanau et al., 2015](https://arxiv.org/abs/1409.0473)
- [Luong et al., 2015](https://arxiv.org/abs/1508.04025)


## Performance
- BLEU score
> "More importantly, the performance of the RNNsearch is as high as that of the conventional phrase-based
translation system (Moses), when only the sentences consisting of known words are considered."
- [BLEU Score paper](https://www.aclweb.org/anthology/P02-1040.pdf)

## Building Training, Eval, and Inference Graphs


## Bidirectional RNNs




[Back to Top](#Table-of-Contents)

# Acknowledgements
- [Genius.com](https://genius.com/)
- DSI instructors: Frank Burkholder, Danny Lumian, Kayla Thomas
- Cohort Peers working with NLP: Matt Devor, Aidan Jared, Lei Shan
- johnwmillr's [LyricsGenius](https://github.com/johnwmillr/LyricsGenius)
- [Gensim's Doc2Vec model](https://radimrehurek.com/gensim/models/doc2vec.html)
- [Robert Meyer's Presentation from PyData's 2017 Berlin Conference](https://www.youtube.com/watch?v=zFScws0mb7M)
- [Andy Jones' blog on the mechanics of Word2Vec](https://andyljones.tumblr.com/post/111299309808/why-word2vec-works)

- [TensorFlow's Seq2Seq](https://github.com/tensorflow/nmt)

## Citations
### Word2Vec Paper
[Mikolov, T., Sutskever, I., Chen, K., Corrado, G. S., & Dean, J. (2013). Distributed representations of words and phrases and their compositionality. In Advances in neural information processing systems (pp. 3111-3119).](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

### Random Walks on Context Spaces Paper
[Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2015). Rand-walk: A latent variable model approach to word embeddings. arXiv preprint arXiv:1502.03520.](https://arxiv.org/abs/1502.03520)

### Doc2Vec Papers
[Le, Q., & Mikolov, T. (2014, January). Distributed representations of sentences and documents. In International conference on machine learning (pp. 1188-1196).](https://arxiv.org/abs/1405.4053)

[Lau, J. H., & Baldwin, T. (2016). An empirical evaluation of doc2vec with practical insights into document embedding generation. arXiv preprint arXiv:1607.05368.](https://arxiv.org/abs/1607.05368)


[Back to Top](#Table-of-Contents)
