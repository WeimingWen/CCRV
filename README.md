# Cross-Lingual Cross-Platform Rumor Verification Pivoting on Multimedia Content

# Overview

This repository contains the implementation of methods in "Cross-Lingual Cross-Platform Rumor Verification Pivoting on Multimedia Content".

# Library Dependencies
  - Python 2.7
  - Pytorch
  - scikit-learn
  - Theano
  - Keras (with Theano backend)
  - Pandas
  - ...

# Data
All data used in this project is save in the 'data' folder. The original Twitter dataset from [VMU 2015](https://github.com/MKLab-ITI/image-verification-corpus) is saved as ‘resources/dataset.txt’ in json format. The additional data we collected from Google and Baidu from VMU 2016 is saved as ‘google_results.txt’ and 'baidu_results.txt'.


# Procedure
1. To reproduce experiments results, simply run main.py

2.	Download parallel English and Mandarin sentence of news and microblogs from [UM-Corpus](http://nlp2ct.cis.umac.mo/um-corpus/index.html) and save them in a folder named 'UM_Corpus'.

3.	Run prepare_UM_Corpus.py to split and tokenize the data in UM-Corpus.

4.	Run train_multilingual_embedding.py to train the multilingual sentence embedding.

5.	Run prepare_FNC_split.py to tokenize, embed and split the data from [Fake News Challenge](http://www.fakenewschallenge.org/).

6.  Run train_agreement_classifier.py to train the agreement classifier.

7.	Run prepare_CCMR.py to tokenize the CCMR dataset.

8.	Run extract_clcp_feats.py to extract all cross-lingual cross-platform features and splits of the data we need for experiments. CLCP saves the available output file.

9. Play with main.py and other scripts to test everything from the Paper.