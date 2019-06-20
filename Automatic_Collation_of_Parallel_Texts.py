#!/usr/bin/env python
# coding: utf-8

# # Automatic Collation of Parallel Texts
# 
# Author: Lucas van der Deijl, University of Amsterdam  
# Version: 20 June 2019  
# Contact: l.a.vanderdeijl@uva.nl, www.lucasvanderdeijl.nl  
# Project: 'Radical Rumours' (Funded by NWO 2017-2021)
# 
# <img src="https://www.nwo.nl/binaries/content/documents/nwo/algemeen/documentation/application/ttw/logos/nwo-eng-cmyk-jpg/NWO+ENG+CMYK+%28JPG%29.jp" style="width:200px; float: left;" />
# 

# ## Aim of this program
# 
# The aim of this program is to automate three tasks: 
# + to align equivalent sentences from a pair of parallel texts*; 
# + to formalise and calculate the relative difference between these equivalent sentences;
# + to extract the lexical differences per aligned sentence-pair.
# 
# *A pair of parallel text variants can consist of two different (mono-lingual) translations of a given source text or of two variants of the same text (a manuscript and its printed equivalent, two different editions of the same text etc.)
# 
# The results of these tasks will be both visualised directly and stored as .png images in the directory of this notebook. In addition, a copy of all results will be stored in a structured .txt-file in the directory of this notebook. 
# 
# This Jupyter notebook can be used to reuse the code or to replicate the collation with your own pair of parallel texts. Run each code block individually or use the 'Run all'-option from the Cell-tab.
# 
# ## Pipeline
# 
# The pipeline desigend to achieve the program's aim performs the following steps:
# + Import the required libraries
# + Install the required libraries (if needed)
# + Load the files
# + Set the parameters
# + Preprocess the texts
# + Create a list of unique words
# + Create chunks ('windows') of a set number of consecutive sentences from both texts
# + Align equivalent windows and sentences
# + Calculate similarity between aligned sentence pairs
# + Visualise the results
# <hr>

# ### Import the required libraries
# 
# First, the required libaries and resources need to be imported. By default the method for tokenization is set to Dutch, but you can change this to any language the nltk supports by replacing 'dutch' for 'english', 'spanish', 'german' etc.:

# In[ ]:


import re
import nltk.data
import string
import itertools
import matplotlib.pyplot as plt
from math import log, pow, sqrt
from scipy import spatial
from nltk.translate.bleu_score import sentence_bleu
from nltk import edit_distance as levenshtein

tokenizer = nltk.data.load('tokenizers/punkt/dutch.pickle') # change 'dutch' in this line to the language of your material


# ## Install the required libraries (if needed)
# 
# In case you got an error after the previous step because not all of the required modules are installed, you can uncomment (remove the '#') the relevant install-command below and run the code. Once the module is installed, run the block above again to import it before moving on to the next step.

# In[ ]:


#!pip install nltk
#!pip install itertools
#!pip install scipy


# ### Set the parameters
# 
# Parameters to be set before analysis
# + source_window_size [default = 10 sentences]
# + target_window_size [default = 16 sentences]
# + alignment_threshold [default = 0.1]

# In[ ]:


source_window_size = 10 # define the window size (number of sentences per chunk) for the source 
target_window_size = 16 # define the window size (number of sentences per chunk) for the target (> source_window_size in order to cope with 1-2 / 1-3 / 1-4 etc. alignments)
alignment_threshold = 0.1 # the minimum cosine similarity for a pair of sentences to be aligned


# ### Load the files
# 
# The next step is to read the two inputfiles. These files should be stored in the directory of this Jupyter notebook and named 'source.txt' and 'target.txt'. Two sample files are available in the directory, containing two different Dutch editions of Descartes's _Discours de la méthode_ (1637). If you prefer to store the files elsewhere or use different filenames, you can edit the filepaths in the code below. Input files need to be stored in .txt and encoded in UTF-8.
# 
# This step also defines a list of stopwords, to be used to visualise the most frequent lexical differences in the final step of this program. A list of stopwords for historical Dutch is included by default. Scholars working with material in a different language can upload their own stopwordsfile in the directory of this notebook and edit the filename in the code below.

# In[ ]:


#set the file paths for the two input documents (defined as the 'source' and the 'target'). 
source = open("source.txt", encoding="UTF-8").read()
target = open("target.txt", encoding="UTF-8").read()

#create a workfile and a log file that stores the sentences the program was unable to match. This is where we will save the output.
workfile = open("workfile.txt", 'w', encoding="UTF-8")
unmatched = open("unmatched_sentences.txt", 'w', encoding="UTF-8")

#define a stopwords list
stopwords = open("stopwordsfile_Dutch.txt", encoding="UTF-8").read().split()

#create headers in the workfile
workfile.write("source window size: %d;target window size: %d ; ; ; ; ; ; ; \n" %(source_window_size, target_window_size))
workfile.write("source_IDs; target_ID; average_similarity; cos_sim; bleu_score; levenshtein_similarity; tokens_in_source_window; tokens_in_target_window; ; ; source; target\n")
unmatched.write("sentence_number; sentence; \n")


# ### Preprocess the texts
# 
# Next, the program prepares both input texts for analysis by removing punctuation and lowercasing all text. It also stores all sentences in a list of sentences for each text, in the following format: 
# + source_list = [['this', 'is', 'the', 'first', 'sentence', 'of', 'source'],['this', 'is', 'the', 'second', 'sentence', 'of', 'source'],[etc.]]
# + target_list = [['this', 'is', 'the', 'first', 'sentence', 'of', 'target'],['this', 'is', 'the', 'second', 'sentence', 'of', 'target'],[etc.]]

# In[ ]:


#create a list of sentences from the source and target files
source_sentences = tokenizer.tokenize(source)
target_sentences = tokenizer.tokenize(target)
    
source_list = []
target_list = []
source_windows = []
target_windows = []

# tokenize, remove punctuation and lowercase the sentences in both the source and the target file
for sentence in source_sentences:
    sen = sentence.lower().translate(str.maketrans("", "", string.punctuation)).encode('ascii', 'ignore').decode('utf-8')
    tokens = nltk.word_tokenize(sen)
    source_list.append(tokens)
print("All sentences from the source have been pre-processed succesfully (tokenization, lowercasing, removal of punctuation)")

for sentence in target_sentences:
    sen = sentence.lower().translate(str.maketrans("", "", string.punctuation)).encode('ascii', 'ignore').decode('utf-8')
    tokens = nltk.word_tokenize(sen)
    target_list.append(tokens)
print("All sentences from the source have been pre-processed succesfully (tokenization, lowercasing, removal of punctuation)")
print(len(target_list), "sentences from the target document have been stored")
print(len(source_list), "sentences from the source document have been stored")


# ### Create chunks ('windows') of a set number of consecutive sentences from both texts
# 
# The next step is to parse the two lists of sentence and merge several adjecent sentences into sentence windows. This step enables 'macro-alignment' of windows of sentences. The size of the windows can be set (see 'Set the parameters' above).

# In[ ]:


# Seperate the list of sentences from the source file into unique (non-overlapping) fragments of source_window_size
i = 0
sen_counter = 0

for no in range(0, len(source_list)):        
    window = []
    if i == (sen_counter + source_window_size):
        sen_counter += source_window_size
        for bla in range(0, source_window_size):
            window += (source_list[((no - source_window_size) + bla)])
        source_windows.append(window)            
    i += 1
print("Source windows created succesfully")
   
# Seperate the list of sentences from the target file into incrementing (overlapping) fragments of target_window_size
   
for no in range(0, len(target_list)):        
    window = []
    if no <= (len(target_list) - target_window_size):
        for sen_in_window in range(0, target_window_size):
            window += (target_list[(no + sen_in_window)])
        target_windows.append(window)
print("Target windows created succesfully")    


# ### Create a list of unique words
# 
# The following block of code creates a list of unique words in the corpus (source + target). It also calculates the document frequencies for each unique word and stores those values in a dictionary to be used for calculating the tf-idf values in the next step.

# In[ ]:


# Create a dictionary of all unique words and their corresponding frequencies in the source files
unique_words = list(set([item for sublist in source_list for item in sublist] + [item for sublist in target_list for item in sublist]))
df_per_word = {}

for word in unique_words:
    if word not in df_per_word:
        df_per_word[word] = 0
    for sentence in (source_list + target_list):
        if word in sentence:
            df_per_word[word] += 1
print("Dictionary of", len(unique_words), "unique words created succesfully")


# ### Align equivalent windows and sentences
# 
# The next step is to align equivalent windows and then equivalent sentences within those windows. The program performs the following steps:
# + for each sentence window from the source document, calculate the relative word frequencies (expressed as tf-idf values) and compare them with the word frequencies in all sentence windows from the target document; 
# 	+ calculate the cosine similarity between the possible pairs of the sentence window and all candidate target windows [source window 1 = target window 1, source window 1 = target window 2, source window 1 = target window 3 etc.];
# 	+ rank the target windows based on cosine similarity in descending order. Define the target window with the highest cosine similarity (rank = 1) as the equivalent of the source window;
#         + for each sentence in the sentence window calculate the cosine similarity between that sentence and all possible candidate sentences from the aligned target window;
#         + rank all sentences from the aligned target window based on cosine similarity in descending order; 
#         + define the target sentence with the highest cosine similarity (ranked first in the previous step) as the most likely equivalent of the source sentence. If none of the potential sentence pairs scored above the alignment_threshold (of 0.1 by default), then the program establishes no match for that sentence: the similarity is considered too low for a reliable match.

# In[ ]:


aligned_sens = {}
unmatched_sents = []

for no in range(len(source_windows)):	# for each window from the source file, identify the equivalent window from the target file
    compare = {}
    i = 0
    for target_window in target_windows: 
        unique_words_windowpair = list(set(source_windows[no] + target_window))
        sourcewindow_vector = []
        targetwindow_vector = []
        for word in unique_words_windowpair:
            occ_sw = source_windows[no].count(word)
            occ_tw = target_window.count(word)
            tf_sw = occ_sw / len(source_windows[no])
            tf_tw = occ_tw / len(target_window)
            df = df_per_word[word]
            idf = log((len(source_windows) +len(target_windows))  / (1 + df))
            tfidf_sw = tf_sw * idf
            tfidf_tw = tf_tw * idf
            sourcewindow_vector.append(tfidf_sw)
            targetwindow_vector.append(tfidf_tw)
        cos_sim_windowpair = 1 - spatial.distance.cosine(sourcewindow_vector, targetwindow_vector)
        compare[i] = cos_sim_windowpair
        i+= 1
    match = False
# sort the dictionary and define the target_window with the highest cosine similarity, based on tf-idf values, as the equivalent window 
    for k,v in sorted(compare.items(), key=lambda compare: compare[1], reverse=True):
        if match == False:
                
# for each sentence from the source_window, find the most similar sentence from the equivalent target window
#create lists of all sentences from the source window and the equivalent target window
            source_win_sentences = []
            target_win_sentences = []
            first_sen_of_source_win = (no * source_window_size)
            first_sen_of_target_win = k
            i = first_sen_of_source_win
            j = first_sen_of_target_win
            for line in itertools.islice(source_list, first_sen_of_source_win, (first_sen_of_source_win + source_window_size)):
                source_win_sentences.append(source_list[i])
                i += 1
            for line in itertools.islice(target_list, first_sen_of_target_win, (first_sen_of_target_win + target_window_size)):
                target_win_sentences.append(target_list[j])
                j += 1
              
            match = True
            sen_no = first_sen_of_source_win

            for source_sen in source_win_sentences:                     # For each sentence x from source_window x
                compare2 = {}
                i = first_sen_of_target_win
                for target_sen in target_win_sentences:
                    unique_words_senpair = list(set(source_sen + target_sen))
                    sourcesen_vector = []
                    targetsen_vector = []
                    for word in unique_words_senpair:
                        occ_ss = source_sen.count(word)
                        occ_ts = target_sen.count(word)				
                        tf_ss = occ_ss / len(source_sen)
                        tf_ts = occ_ts / (len(target_sen) + 1) # + 1 to avoid ZeroDivisionError
                        df = df_per_word[word]
                        idf = log((len(source_list) + len(target_list)) / (1 + df))
                        tfidf_ss = tf_ss * idf
                        tfidf_ts = tf_ts * idf
                        sourcesen_vector.append(tfidf_ss)
                        targetsen_vector.append(tfidf_ts)
                    cos_sim_senpair = 1 - spatial.distance.cosine(sourcesen_vector, targetsen_vector)
                    compare2[i] = cos_sim_senpair                        
                    i += 1
                match2 = False
                for k,v in sorted(compare2.items(), key=lambda compare2: compare2[1], reverse=True):
                    if match2 == False:
                        if v > alignment_threshold:
                            if k not in aligned_sens:
                                aligned_sens[k] = []
                            aligned_sens[k].append(sen_no)
                               
                            print("Match found between sentence ", sen_no, " and ", k, "confidence:", v)
                            sen_no += 1
                            match2 = True

                if match2 == False:                        
                    print("no matching sentence was found for sentence ", sen_no)
                    unmatched_sents.append((sen_no, source_sen))
                    unmatched.write("%d; %s; \n" %(sen_no, source_sen))
                    sen_no += 1
print("Sentence alignment complete")
print((len(source_list) - len(unmatched_sents)), "/", len(source_list), "sentences were aligned succesfully (= ", ((len(source_list) - len(unmatched_sents)) / len(source_list)), "%)")
unmatched.close()


# ### Calculate similarity between aligned sentence pairs
# 
# Once the program has aligned all equivalent sentences, it will calculate the lexical overlap and extract the lexical differences between the sentences of each sentence pair. For each sentence pair, it:
# 
# + calculates the BLEU-score, the cosine similarity and the Levenshtein distance (normalised and converted to a similarity metric) between the equivalent sentences;
# + extracts all unique words from the two equivalent sentences;
# + saves both sentences from the aligned sentence pair, the unique words in the source and in the target, the cosine similarity, the BLEU-score and the Levensthein distance in the workfile.

# In[ ]:


#calculate and save metrics for all aligned sentence-pairs

xvalues = []
BLEU_yvalues = []
Levenshtein_yvalues = []
Cosine_yvalues = []

untrans_in_source_total = []
untrans_in_target_total = []

for target_index in aligned_sens:
    xvalues.append(target_index)
    source_indices = ','.join(str(e) for e in aligned_sens[target_index])
    source_sen = []
    target_sen = target_list[target_index]
    for sen in aligned_sens[target_index]:
        source_sen += source_list[sen]

#calculate Bleu-score
    reference = []
    reference.append(source_sen)
    bleu = sentence_bleu(reference, target_sen)
    BLEU_yvalues.append(bleu)
    
#calculate relative number of overlapping words
    unique_words_senpair = []
    untransl_insource = []
    untransl_intarget = []
    for word in source_sen:
        if word not in target_sen:
            untransl_insource.append(word)
        if word not in unique_words_senpair:
            unique_words_senpair.append(word)
    for word in target_sen:
        if word not in source_sen:
            untransl_intarget.append(word)
        if word not in unique_words_senpair:
            unique_words_senpair.append(word)
    similarity_source = 1 - (len(untransl_insource) / len(source_sen))
    similarity_target = 1 - (len(untransl_intarget) / len(target_sen))
    average_sim = (similarity_source + similarity_target) / 2
    untrans_in_source_total.append(untransl_insource)
    untrans_in_target_total.append(untransl_intarget)

#calculate cosine similarity 
    source_vector = []
    target_vector = []
    for word in unique_words_senpair: 
        tf_s = source_sen.count(word) / len(source_sen)
        tf_t = target_sen.count(word) / len(target_sen)
        df = df_per_word[word]
        idf = log((len(source_list) + len(target_list)) / (1 + df))
        tfidf_s = tf_s * idf
        tfidf_t = tf_t * idf
        source_vector.append(tfidf_s)
        target_vector.append(tfidf_t)
    cos_sim = 1 - spatial.distance.cosine(source_vector, target_vector)
    Cosine_yvalues.append(cos_sim)

#calculate Levenshtein-distance
    source_str = ' '.join(source_sen)
    target_str = ' '.join(target_sen)
    if len(source_sen) >= len(target_sen):
        levenshtein_sim = 1- sqrt(pow((levenshtein(source_str, target_str) / len(source_sen)), 2))
    else:
        levenshtein_sim = 1- (levenshtein(source_str, target_str) / len(target_sen))
    Levenshtein_yvalues.append(levenshtein_sim)
    
    #print and save the results		
    print("%s; %d; %g; %g; %g; %g; %d; %d; \n" %(source_indices, target_index, average_sim, cos_sim, bleu, levenshtein_sim, len(source_sen), len(target_sen)))
    workfile.write("%s; %d; %g; %g; %g; %g; %d; %d; %s; %s; %s; %s; \n" %(source_indices, target_index, average_sim, cos_sim, bleu, levenshtein_sim, len(source_sen), len(target_sen), untransl_insource, untransl_intarget, source_str, target_str))
print("All results have been saved succesfully in the workfile")
workfile.close()


# ### Visualise the results
# 
# The visualisations below display:
# + the similarity for each aligned sentence pair using two different similarity measures: BLEU and cosine similarity;
# + the most frequent lexical differences between all aligned sentence pairs.
# 
# Both visualisations will be stored as .png-files in the directory of this notebook. You can change the title of the graphs in the code below.

# #### Similarity by aligned sentence pair

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig = plt.figure(figsize=(8,4), dpi=300)

plt.plot(range(0,len(xvalues)), BLEU_yvalues, label="BLEU-4", linewidth=0.4)
plt.plot(range(0,len(xvalues)), Cosine_yvalues, label="Cosine", linewidth=0.4)

plt.style.use("default")
#plt.plot(xvalues, Levenshtein_yvalues)
plt.suptitle("Similarity between 'source.txt' and 'target.txt' by sentence pair")
plt.xlabel("Sentence pair")
plt.ylabel("Similarity")
plt.legend()
plt.show()

fig.savefig("similarity_plot.png") #save the file


# #### Most frequent lexical differences between 'source' and 'target'

# In[ ]:


untrans_in_source_flattened = [item for sublist in untrans_in_source_total for item in sublist if item not in stopwords]
untrans_in_source_freqs = nltk.FreqDist(untrans_in_source_flattened)

fig = plt.figure(figsize=(8,4), dpi=300)

untrans_in_target_flattened = [item for sublist in untrans_in_target_total for item in sublist if item not in stopwords]
untrans_in_target_freqs = nltk.FreqDist(untrans_in_target_flattened)
untrans_in_source_freqs.plot(40,title="Lexical differences between 'source.txt' and 'target.txt'")
#untrans_in_target_freqs.plot(40,title="Lexical differences between 'source.txt' and 'target.txt'")

fig.savefig("lexical_differences_plot.png") #save the file

