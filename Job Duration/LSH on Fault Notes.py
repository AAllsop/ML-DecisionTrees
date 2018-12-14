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

def generate_hash_calc (hash_calc_iter,col_name,modulo_no,working_df) :
#    hash_calc_iter= 1      
#    col_name = 'h1'
    working_df[col_name] = hash_lookup(hash_calc_iter,working_df.index,modulo_no)
    
    #create a boolean mask to assist in getting the modulo value
    #   index has to be the same across the mask table and it's target
    band2 = working_df.iloc[:,0:1]
    band2 = band2.drop(['index'],axis=1)
    frame_col_count = len(working_df.columns)
    band_hashs = pd.concat([working_df[col_name]]*frame_col_count,axis=1)
    band_hashs.columns = working_df.columns
    band3 = [band2,band_hashs]
    band4 = pd.concat(band3,sort=True,axis=1)
    mask = working_df==1
    modulo_values_pd = band4[mask].fillna(999999).astype(int)
    modulo_values_pd = modulo_values_pd.drop(columns=["band",col_name,"index"])
    #bring back the band
    modulo_values_pd = pd.merge(modulo_values_pd,working_df.loc[working_df.index,"band"].to_frame(),left_index=True,right_index=True,how="inner")
    hash_values_pd = modulo_values_pd.groupby(["band"]).min()
    return hash_values_pd 


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
fault_notes_stg = fault_notes_dz.copy().head(10000)

cut_extent = 40
fault_notes_stg["partition"] = pd.qcut(fault_notes_stg.index,cut_extent,labels=False,retbins=False)

#cycle through y% partitions (training data)
#compiled_docs = None
end_rng = int(cut_extent*0.75) + 1
partition_range = range(1,end_rng)
for partition in partition_range:
#    partition = 1
    fault_notes = fault_notes_stg[fault_notes_stg["partition"] == partition]
   
    #vectorize shingles
    partition_cols,vector =  vectorise_shingles(fault_notes["FaultNotes_Cleaned"])
    shingle_matrix = pd.DataFrame(vector.toarray(), columns=partition_cols)
    shingle_matrix.index = fault_notes.index
    shingle_frame = shingle_matrix.transpose().reset_index()
    shingle_frame  = pd.merge(shingle_frame ,cols_pd,left_on=["index"],right_on =["shingles"],how="inner")
    shingle_frame .index = shingle_frame ["shingle_index"]
    shingle_frame  = shingle_frame.drop(columns=["shingles","shingle_index"])
    
    hash_values_stack_all = None
    hash_calcs = range(1,11)
    for hash_calc in hash_calcs:
#        hash_calc  =2 
        hash_col_name = generate_hash_col_name(hash_calc)
        
        #get the modulo no. which is the max count of rows in each band
        modulo_no = max(cols_pd.groupby("band")["shingle_index"].count())
        
        hash_values = generate_hash_calc(hash_calc,hash_col_name,modulo_no,shingle_frame)
        
        hash_values_stack = hash_values.stack().to_frame()
        index = pd.MultiIndex.from_tuples(hash_values_stack.index,names = ["band","doc"])
        hash_values_stack.index = index
        hash_values_stack = hash_values_stack.rename(columns={0:hash_col_name})

        if hash_values_stack_all is None:
            hash_values_stack_all = pd.DataFrame(index=hash_values_stack.index)
        
        hash_values_stack_all = pd.merge(hash_values_stack_all,hash_values_stack,left_index=True,right_index=True,how="inner")

    #Calculate the hash signature    
    hash_values_stack_all = hash_values_stack_all[hash_values_stack_all[hash_col_name] != 999999]
    hash_values_stack_all["signature"] = hash_values_stack_all.index.map(dict(zip(hash_values_stack_all.index,pd.Series(hash_values_stack_all.fillna('').values.tolist()).map(lambda x : '.'.join(map(str,x))))))
    
    #Export doc hash signatures
    hash_values_stack_all = pd.merge(hash_values_stack_all,fault_notes[["FaultID"]],left_on = hash_values_stack_all.index.get_level_values(1), right_index=True,how="inner")
    hash_values_stack_all.to_csv("Data\LSH Outputs\doc_hash_signatures_" + str(partition) + ".csv",columns=["FaultID","signature"])
    
#Export shingles
cols_pd.to_csv("Data\LSH Outputs\shingles.csv")
    

    
    
    
    
    
    
    
    
    
    
    
    
            
            
            
            