import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import stop_words 
    
def hash_lookup(f,x,mod_op):
    #f = a number to use as an offset, x = row index, mod_op is the modulo number    
    return np.mod(f*x+1,mod_op) 

def vectorise_shingles (doc):
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(doc)
    cols = vectorizer.get_feature_names()
    result = [cols]
    result.append(X)
    return result

def generate_hash_col_name (hash_calc_iter):
    hash_col_name = "h" + str(hash_calc_iter)
    return hash_col_name

def generate_hash_calc (hash_calc_iter,col_name) :
#    hash_calc_iter= 3      
    band[col_name] = hash_lookup(hash_calc_iter,band.index,len(band))
    #create a boolean mask to assist in getting the modulo value
    #   index has to be the same across the mask table and it's target
    band2 = band.iloc[:,0:1]
    band2 = band2.drop(['index'],axis=1)
    band_col_count = len(band.columns)
    band_hashs = pd.concat([band[col_name]]*band_col_count,axis=1)
    band_hashs.columns = band.columns
    band3 = [band2,band_hashs]
    band4 = pd.concat(band3,sort=True,axis=1)
    mask = band==1
    modulo_values_pd = band4[mask].fillna(9999).astype(int)
    modulo_values = pd.DataFrame.from_dict(dict(zip(modulo_values_pd.columns,modulo_values_pd.min(axis=0) )),orient="index")  
    modulo_values = modulo_values.rename(columns={modulo_values.columns[0]:col_name})
    return modulo_values


stop = list(stop_words.ENGLISH_STOP_WORDS)

#import fault notes
fault_notes_dz = pd.read_csv("Data\FaultData.csv", low_memory=False)

#remove stop words from notes
fault_notes_dz["FaultNotes_Cleaned"] = fault_notes_dz["FaultNotes"].map(lambda x: ' '.join([word for word in str(x).lower().split() if word not in (stop)]))

#create a word vector and create a banded frame of shingles
cols = vectorise_shingles(fault_notes_dz["FaultNotes_Cleaned"])[0]
cols_pd = pd.DataFrame(cols,columns=["shingles"])
cols_pd["shingle_index"] = cols_pd.index
cols_pd["band"] = pd.qcut(cols_pd.index,100,labels=False,retbins=False)

#split into train/test
#split frame in x partitions
fault_notes_stg = fault_notes_dz.copy().head(1000)

cut_extent = 40
fault_notes_stg["partition"] = pd.qcut(fault_notes_stg.index,cut_extent,labels=False,retbins=False)

#cycle through y% partitions (training data)
end_rng = int(cut_extent*0.75) + 1
partition_range = range(1,end_rng)
for partition in partition_range:
#    partition = 1
    fault_notes = fault_notes_stg[fault_notes_stg["partition"] == partition]
    
    #vectorize shingles
    partition_cols,vector =  vectorise_shingles(fault_notes["FaultNotes_Cleaned"])
    
    shingle_matrix = pd.DataFrame(vector.toarray(), columns=partition_cols)
    shingle_frame = shingle_matrix.transpose().reset_index()
    shingle_frame  = pd.merge(shingle_frame ,cols_pd,left_on=["index"],right_on =["shingles"],how="inner")
    shingle_frame .index = shingle_frame ["shingle_index"]
    shingle_frame  = shingle_frame .drop(columns=["shingles","shingle_index"])
    
    signatures_pd = pd.DataFrame(columns=["band","signature"])
    
    hash_calcs = range(11)
    for band_no in np.unique(shingle_frame["band"]):
        band_no = 1
        global band
        band = shingle_frame [shingle_frame["band"]==band_no]

        band_hash_signatures_pd = pd.DataFrame(index=band.columns)
        band_hash_signatures_pd["band"] = band_no
        
        for hash_calc in hash_calcs:
            hash_calc = hash_calc +1
            hash_col_name = generate_hash_col_name(hash_calc)
            mod_values = generate_hash_calc(hash_calc,hash_col_name)
            
            band_hash_signatures_pd = pd.merge(band_hash_signatures_pd,mod_values,left_index=True,right_index=True, how="inner")
        band_hash_signatures_pd["signature"] = band_hash_signatures_pd.index.map(dict(zip(band_hash_signatures_pd.index,pd.Series(band_hash_signatures_pd.fillna('').values.tolist()).map(lambda x : '.'.join(map(str,x))))))
        band_hash_signatures_pd = band_hash_signatures_pd[band_hash_signatures_pd[hash_col_name] != 9999]
            
        frames = [signatures_pd,band_hash_signatures_pd[["band","signature"]]]
        signatures_pd = pd.concat(frames)
    
    
    
    
    
    
        d = signatures_pd.groupby(["signature"])["signature"].count().to_frame()
        d = d[d["signature"]>1]
    
    










def insert_space_character (a,n):
    #a = initial string, i = insert a space every n characters
    return ' '.join([a[i:i + n] for i in range(0,len(a),n)])

#fault_notes["FaultNotes_Cleaned"] = fault_notes["FaultNotes_Cleaned"].map(lambda x: insert_space_character(x,8))
    


#    vectorizer = CountVectorizer(binary=True)
#    X = vectorizer.fit_transform(fault_notes["FaultNotes_Cleaned"])
#    partition_cols = vectorizer.get_feature_names()
#    
    

            #hash_calc = 3      
#            hash_col_name = "h" + str(hash_calc)
#            band[hash_col_name] = hash_lookup(hash_calc,band.index,len(band))
#            
#            #create a boolean mask to assist in getting the modulo value
#            #   index has to be the same across the mask table and it's target
#            band2 = band.iloc[:,0:1]
#            band2 = band2.drop(['index'],axis=1)
#            
#            band_col_count = len(band.columns)
#            band_hashs = pd.concat([band[hash_col_name]]*band_col_count,axis=1)
#            band_hashs.columns = band.columns
#            band3 = [band2,band_hashs]
#            band4 = pd.concat(band3,sort=True,axis=1)
#            mask = band==1
#            modulo_values_pd = band4[mask].fillna(9999).astype(int)
#            
#            modulo_values = pd.DataFrame.from_dict(dict(zip(modulo_values_pd.columns,modulo_values_pd.min(axis=0) )),orient="index")  
#            modulo_values = modulo_values.rename(columns={modulo_values.columns[0]:hash_col_name})
#            