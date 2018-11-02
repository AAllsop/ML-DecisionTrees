#function accepts a pandas dataframe of a single column
#   assumes an index on the df

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words 

workpath = r"C:\Users\allsopa\OneDrive - City Holdings\Development\Development Tasks\20181001_ML for Job Duration" + "\\"

stop = list(stop_words.ENGLISH_STOP_WORDS)

#import fault notes
fault_notes = pd.read_csv(workpath + "PreAnalysis - FaultNotes.csv")
fault_notes = fault_notes.rename(columns=({"Unnamed: 0":"Notes"}))
text_dataframe = fault_notes
text_dataframe = text_dataframe .rename(columns=({"Notes":"Text"}))

#remove stop words from notes
text_dataframe["Text"] = text_dataframe["Text"].map(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))

#split dataset to allow processing to occur within memory limits
#   define length of each chunk
chunk_split_on = np.ceil(len(text_dataframe)/1000) 

def aggregate_word_counts (ngram):
    #initialise a dataframe to store results
    text_words = pd.DataFrame()
#    for chunk in np.array_split(text_dataframe,chunk_split_on):
        chunk = np.array_split(text_dataframe,chunk_split_on)[1]
        vectorizer = CountVectorizer(stop_words="english",ngram_range=(1,1))
        X = vectorizer.fit_transform(chunk["Text"])
        vocab = vectorizer.get_feature_names()    
        counts = X.toarray().sum(axis=0)
        d = pd.DataFrame(X.toarray(), columns = vocab)
        
        d2 = d[d["access"] == 1]
        
        vocab_counts = pd.DataFrame(pd.Series(dict(zip(vocab,counts))))
        #append vocab and counts to main dataframe
        text_words = text_words.append(vocab_counts)
    text_words = text_words.reset_index()        
    text_words = text_words.rename(columns={"index":"vocab",0:"Frequency"})
    text_word_counts = text_words.groupby("vocab")["Frequency"].sum().reset_index()
    return text_word_counts

#produce files based on the ngram of a bag of words
text_word_counts_1 = aggregate_word_counts(1)
text_word_counts_2 = aggregate_word_counts(2)
text_word_counts_3 = aggregate_word_counts(3)

text_word_counts_1.to_csv("text_word_counts_1.csv")
text_word_counts_2.to_csv("text_word_counts_2.csv")
text_word_counts_3.to_csv("text_word_counts_3.csv")

