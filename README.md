# CS 410 Course Project - Michael McClanahan (mjm31)
## Reproduction of Paper

### Mining Causal Topics in Text Data: Iterative Topic Modeling with Time Series Feedback (Kim et. al. 2013)
[Link to paper](https://dl-acm-org.proxy2.library.illinois.edu/doi/pdf/10.1145/2505515.2505612)

## Overview of Implementation

This project attempts to reproduce the 2020 Presidential Election experiment outlined in the paper above.

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
- ```cleaned_doc_list```: This the document collection to be analyzed (ie: a list of documents represented each as a list of tokens or words).
- ```gensim_dictionary```: This is the gensim dictionary object to be analyzed (built from gore_bush_doc_list).
- ```gensim_corpus```: This is the gensim corpus to be analyzed (ie: a list of documents represented each as a list of wordIDs and their counts in the document)
- ```word_impact_dict```: This is a dictionary of corpus {wordID:(impact score,p-value)}.  It represents the result of section 4.2.2's Word-level Causality analysis.

At runtime, if the script's ```reload_data``` variable is set to ```False```, the script will reload ```president_norm_stock_ts``` and ```gore_bush_nyt_ts``` from disk in O(1) time.  If set to ```True```, functions ```build_datasets()``` and ```parse_nyt_corpus_for_gore_bush()``` will get called and rebuild these datasets from a .csv file and the NYT corpus for XML documents, respectively.  Since the NYT dataset was too large, it was not uploaded to this repository.  Therefore, setting this variable to ```True``` is not recommended.

Additionally, if the script's ```build_new_corpus``` variable is set to ```False```, it will reload all of the other remaining objects from disk in O(1) time.  If set to ```True```, it will rebuild all of the other objects by rebuilding the collection, the [gensim dictionary](https://radimrehurek.com/gensim/corpora/dictionary.html), and the gensim corpus (which is the object passed to the [LdaModel()](https://radimrehurek.com/gensim/models/ldamodel.html) for its  ```corpus``` parameter.  It will then reperform the Word-level Causality Analysis from 4.2.2, storing the result per gensim dictionary word ID in a dictionary for quick lookup during ITMTF iterations.

The following 4 parameters are then set and ITMTF iterations are started by calling the ```ITMTF()``` recursive function.
```
min_significance_value = 0.8
min_topic_prob = 0.001
iterations = 5
number_of_topics = 10
causal_topics = ITMTF(gore_bush_gensim_corpus,gore_bush_gensim_dictionary,number_of_topics,number_of_topics,word_impact_dict,gore_bush_nyt_ts,president_norm_stock_ts,ts_tsID_map,min_significance_value,min_topic_prob,iterations)
```

The ITMTF function will call itself for the number of iterations specified, each time building an [LdaModel() object](https://radimrehurek.com/gensim/models/ldamodel.html), passing in a 2D matrix (num_topics,num_unique_terms) matrix of re calculated prior topic word probability distributions into its ```eta``` parameter. To re-calculate the prior, causal topics are first extracted during the topic-level causality analysis (see section 4.2.1 in the paper), which is performed using a Pearson correlation against the reference Topic Stream and the presidential stock price time series. The topic-word probability prior is then calculated for top words in causal topics in the ```build_prior_matrix()``` function.  This function makes use of the ```word_impact_dict``` object above for each causal topic. 

With each iteration, .csv files ```causal_topic_words.csv``` and ```itmtf_stats.csv``` in the working directory are appended with a list of signficant topics and their top 5 words as well as that iteration's average causality confidence and average purity, respectively.

### Goal

The project aims to reproduce the paper's results documented in Table 2 and Figure 2(b).  You will note that the µ parameter from the paper is not defined prior to calling the ITMTF function above, primarily because it is not an parameter for the LdaModel() class provided by Gensim.  It is for this reason that Figure 2(a) from the paper was not reproduced.

## How to Use

1. Install the most recent versions of the above non-standard libraries using pip in a Python3 environment.  Ex:
```pip install pandas```
2. Clone the repository, which contains the working directory and all dependent objects.
3. Navigate to the working directory and run the script with ```python ITMTFPresidential.py```

![example_run](https://github.com/MM026184/CourseProject/blob/main/images/example_run.png)
