import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words 

stop = list(stop_words.ENGLISH_STOP_WORDS)

#import fault notes
fault_notes = pd.read_csv("PreAnalysis - FaultNotes.csv")
fault_notes = fault_notes.rename(columns=({"Unnamed: 0":"Notes_Original"}))

#remove stop words from notes
fault_notes["Notes"] = fault_notes["Notes_Original"].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop)]))

#initialise a dataframe to store results
fault_note_words = pd.DataFrame()

fault_notes_temp = fault_notes.nsmallest(10,columns="OnsiteTimeMinutes")

#split dataset to allow processing to occur within memory limits
#   define length of each chunk
chunk_split_on = np.ceil(len(fault_notes)/1000) 

for chunk in np.array_split(fault_notes,chunk_split_on):
#chunk = np.array_split(fault_notes_temp,chunk_split_on)[0]
    vectorizer = CountVectorizer(stop_words="english",ngram_range=(,1))
    X = vectorizer.fit_transform(chunk["Notes"])
    vocab = vectorizer.get_feature_names()    
    counts = X.toarray().sum(axis=0)
    vocab_counts = pd.DataFrame(pd.Series(dict(zip(vocab,counts))))
    #append vocab and counts to main dataframe
    fault_note_words = fault_note_words.append(vocab_counts)
    
fault_note_words = fault_note_words.reset_index()    
fault_note_words  = fault_note_words.rename(columns={"index":"vocab",0:"Frequency"})
fn_grp = fault_note_words.groupby("vocab")["Frequency"].sum().reset_index()


v = CountVectorizer(ngram_range=(1,1))
print(v.fit(["an apple a day keeps the doctor away"]).vocabulary_)

