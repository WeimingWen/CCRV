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
Three sub-datasets of our CCMR dataset are saved in the folder CCMR
as three json files (lists of json objects), "CCMR/CCMR_Twitter.txt", "CCMR_Google.txt" and "CCMR_Baidu.txt".

For CCMR Twitter, each tweet is saved as a json object with keys "tweet_id", "content", "image_id", "event", "timestamp".
For CCMR Google and Baidu, each webpage is saved as a json object with keys "url", "title", "image_id", "event".

# Procedure
1. To reproduce experiments results, simply run main.py.

2.	Download parallel English and Mandarin sentence of news and microblogs from [UM-Corpus](http://nlp2ct.cis.umac.mo/um-corpus/index.html) and save them in a folder named 'UM_Corpus'.

3.	Run prepare_UM_Corpus.py to split and tokenize the data in UM-Corpus.

4.	Run train_multilingual_embedding.py to train the multilingual sentence embedding.

5.	Run prepare_FNC_split.py to tokenize, embed and split the data from [Fake News Challenge](http://www.fakenewschallenge.org/).

6.  Run train_agreement_classifier.py to train the agreement classifier.

7.	Run prepare_CCMR.py to tokenize the CCMR dataset.

8.	Run extract_clcp_feats.py to extract all cross-lingual cross-platform features and splits of the data we need for experiments. CLCP saves the available output file.

9. Play with main.py and other scripts to test everything from the Paper.