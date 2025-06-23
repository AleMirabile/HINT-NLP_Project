# HINT-NLP_Project
# NLPexam

# HINT - Headline Intelligent News Tagger**Exam Date: June 2024**

In this NLP exam project, I tackle the challenge of developing machine learning classifiers capable of categorizing newspaper headlines into four distinct classes. I gave more importance to experimenting different models than achieving the best results possible, which are, by the way, bounded by the computational resources.


## Table of Contents

- [Introduction](#introduction)
- [Data](#data)
- [Methodology](#methodology)
- [Models](#models)
- [Results](#results)
- [Project Recap](#project-recap)


## Introduction

In this NLP exam project, my project's goal is to classify newspaper headlines into four distinct categories, focusing on comparing various models and exploring their behavior on short, context-limited texts.

## Data

The dataset used is the AG News Corpus. For this project, I extracted a balanced subset of 7,600 headlines (25% per class), splitting it into 6,080 training and 1,520 test samples. Headlines average about 37 words before preprocessing. Since the classes were balanced no further techniques were required.

## Methodology

I follow a structured methodology for my NLP project, including:
Stratified train/validation/test splits to ensure balanced classes

-Text cleaning: lowercasing, stopword/punctuation removal, number stripping, whitespace normalization

-Tokenization via NLTK’s wordpunct_tokenize and stemming with Porter Stemmer

-Construction of sparse TF–IDF matrices from Bag-of-Words

-Transformer tokenization using Hugging Face for BERT and GPT-2

-Prompt-based framing for GPT-2, turning classification into a language-model completion task
## Models

Our project explores the performance of three distinct models:

Three distinct models were explored:

**TF-IDF + Logistic Regression:**
Classical machine learning pipeline leveraging lexical features, sparse representations, and standard preprocessing techniques to handle text noise effectively.

**Pretrained Encoder-only BERT:**
A pretrained bidirectional transformer encoder (BERT) fine-tuned without extensive preprocessing, utilizing contextual embeddings for improved understanding and classification accuracy.

**Prompt-Based Fine-tuned GPT-2:**
A generative GPT-2 model fine-tuned specifically for classification tasks using a prompt-based approach. Initially tested via zero-shot prompting, this method required further fine-tuning to achieve acceptable performance.

## Results

Results have been classified using different metrics and matrices: Accuracy, Precision, Recall, F1-score, Loss(training vs. validation curves) also Bag-of-Words heatmap, Confusion Matrix, ROC Curves and AUC. As expected BERT has been the best performing model.



## Project Recap

In summary, our project involved several key steps.

I tested three headline‐classification techniques on the AG News corpus, each revealing distinct strengths and weaknesses. First, a TF–IDF + Logistic Regression pipeline—complete with full preprocessing techniques, tokenization, and Porter stemming. Next, a fine-tuned bert-base-uncased model, trained without manual cleanup but with a slower learning rate and extra epochs, pushed performance higher. Finally, GPT-2 used in a zero-shot, prompt-based setup fell short, confirming that generative models require task-specific tuning for reliable classification. However I then discarded the zero-shot prompting to try a prompt-based and fine-tuned approach and I got satisfying outcomes. In conclusion, I compared all three with accuracy, precision, recall, F₁, and visualized their behavior via learning curves, confusion matrices, ROC/AUC and loss plots.
