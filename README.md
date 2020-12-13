# CS 410 Course Project - Michael McClanahan (mjm31)
## Reproduction of Paper

### Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback (Kim et. al. 2013)

## Overview of Implementation

All programming was done in Python 3.6 (specifically Anaconda distribution 4.4.0).

A requirements.txt file is provided outlining all non-standard libraries utilized and their associated versions.

```
pandas==1.1.5
gensim==3.8.0
nltk==3.4.4
scipy==1.5.4
numpy==1.19.4
```

All source code is contained within ITMTFPresidential.py as a series of functions.

For convenience, the following objects have been serialized to files (specifically .pkl files) for easy re-use:

- ```president_norm_stock_ts``` : This is the non-text time series containing normalized presidential stock prices (May through October 2020 for the Bush-Gore 2000 presidential race.
- ```gore_bush_nyt_ts``` : This is the document time series containing NY Times articles from May through October 2020 mentioning either Bush or Gore.
- ```gore_bush_doc_list```: This the document collection to be analyzed (ie: a list of documents represented each as a list of tokens or words).
- ```gore_bush_gensim_dictionary```: This is the gensim dictionary object to be analyzed (built from gore_bush_doc_list).
- ```gore_bush_gensim_corpus```: This is the gensim corpus to be analyzed (ie: a list of documents represented each as a list of wordIDs and their counts in the document)
- ```word_impact_dict```: This is a dictionary of corpus {wordID:(impact score,p-value)} 
