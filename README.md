## NC-BERT: A Numerical-Contextual BERT for Numerical Reasoning

This repostiory contains codes for **NC-BERT: Exploiting Numerical-Contextual Knowledge to Improve Numerical Reasoning in Question Answering**.

**NC-BERT** is a numerical reasoning QA model that handles discrete reasoning (e.g., addition, subtraction, counting) to answer a question based on the given passage.

The task at hand is DROP, a numerical question answering dataset created by AllenNLP.

Our model leverages a novel attention masking scheme (namely, the **NC-Mask**) to:
1. Reduce the over-reliance on the parametric knowledge by induceing the model leverage number-related contextual knowledge.
2. And thereby enable the model to correctly interpret the numbers in the passage (consequently improving the numerical reasoning performance).

We also provide the code for pre-training the `ALBERT-xxlarge-v2` model as the initial backbone of the **NC-BERT** model (in this case, the **NC-ALBERT**).

The **NC-ALBERT** model, unlike its BERT counterpart, is trained using the sentence order prediction (SOP) task along with the masked language modeling (MLM) task (Lan et al., 2019).

**Note**
- The sentence order prediction is not implemented on the "sentence-level," but on the "text chunk-level."

### Structure
The repository contains:
* Implementation/pre-training/finetuning of NC-BERT on MLM/synthetic-data/DROP/SQuAD (in `pre_training` dir)
* Code and vocabularies for textual data generation (in `textual_data_generation` dir)
* Code for numerical data generation (in `pre_training/numeric_data_generation` dir)   

Instructions for downloading data + models for pre-trained baseline are in the README of `pre_training` dir.

This repository is based on Geva's [repository](https://github.com/ag1988/injecting_numeracy).