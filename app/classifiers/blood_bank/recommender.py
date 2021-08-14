import re
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity  
from sklearn.metrics import pairwise_distances
from pathlib import Path

class Recommender:
    def __init__(self):
        self.path_to_artifacts = "app/classifiers/blood_bank/saved_data/"
        self.index=["district","blood_bank","blood_bank_address","contact_person","contact_number"]
        self.blod_lib_csv = pd.read_csv(self.path_to_artifacts+'infer_dataset.csv', names=self.index, header=None)
        #load saved classifier model, including its weights and the optimizer
        self.model = self.path_to_artifacts+'cosine_logic'

    def load_logic(self, model):
        infile = open(model,'rb')
        cosine_sim = pickle.load(infile)
        infile.close()
        return cosine_sim

    def preprocessing(self, dataset):
        indices_district = pd.Series(dataset.index, index=dataset['district']).drop_duplicates()
        return indices_district
    
    def recommender_predict(self, district_name, indices_district,  cosine_sim, data):
        idx = indices_district[district_name][0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        location_indices = [i[0] for i in sim_scores]
        return data['blood_bank'].iloc[location_indices]

    def postprocessing(self,prediction):
        dfx = {'BLOOD_BANK': prediction}
        df = pd.DataFrame(dfx)
        df.reset_index(drop=True, inplace=True)
        predictions =df
        return predictions


    def compute_prediction(self, input_data):
        try:
            model = self.load_logic(self.model)
            district_data = self.preprocessing(self.blod_lib_csv)
            prediction = self.recommender_predict(input_data, district_data, model,self.blod_lib_csv)
            prediction = self.postprocessing(prediction)
        except Exception as e:
            return {"status": "Prediction Error", "message": str(e)}

        return prediction
