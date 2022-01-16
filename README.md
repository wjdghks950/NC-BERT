## NC-BERT: A Numerical-Contextual BERT for Numerical Reasoning

This repostiory contains codes for NC-BERT: Exploiting Numerical-Contextual Knowledge to Improve Numerical Reasoning in Question Answering.

It's a novel attention-masking scheme to relieve the over-reliance on the parametric knowledge, and induce the model to leverage number-related contextual knowledge in numerical reasoning over text in DROP.


### Structure
The repository contains:
* Implementation/pre-training/finetuning of GenBERT on MLM/synthetic-data/DROP/SQuAD (in `pre_training` dir)
* Code and vocabularies for textual data generation (in `textual_data_generation` dir)
* Code for numerical data generation (in `pre_training/numeric_data_generation` dir)   

Instructions for downloading data + models for pre-trained baseline are in the README of `pre_training` dir.
