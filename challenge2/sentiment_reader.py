import codecs
import numpy as np
import os
import spacy


class SentimentCorpus:
    
    
    def __init__(self):
        '''
        prepare dataset
        1) build feature dictionaries
        2) split data into train/dev/test sets 
        '''
        path = "20news-bydate-train"
        X, y, feat_dict, feat_counts = self.build_dicts(path)
        self.train_nr_docs = y.shape[0]
        self.train_nr_features = X.shape[1]
        self.train_X = X
        self.train_y = y
        self.train_feat_dict = feat_dict
        self.train_feat_counts = feat_counts
        
        path = "20news-bydate-test"
        X, y, feat_dict, feat_counts = self.build_dicts(path)
        self.test_nr_docs = y.shape[0]
        self.test_nr_features = X.shape[1]
        self.test_X = X
        self.test_y = y
        self.test_feat_dict = feat_dict
        self.test_feat_counts = feat_counts
          
    def build_dicts(self, path):
        nlp = spacy.load("en_core_web_sm")
        feat_counts = {}
        # build feature dictionary with counts'
        rec_autos_dir =  path + "/rec.autos/"
        hardware_dir = path + "/comp.sys.mac.hardware/"
    
        nr_rec = 0
        rec_dir = os.listdir(rec_autos_dir)
        for file in rec_dir :
            with codecs.open(rec_autos_dir + file , 'r') as rec_file:
                for line in rec_file:
                    doc = nlp(line)
                    for tok in doc : 
                        name = tok.text.lower()
                        if name not in feat_counts:
                            feat_counts[name] = 0
                        feat_counts[name] += 1
                nr_rec += 1
        
                            
        nr_hard = 0
        hard_dir = os.listdir(hardware_dir)
        for file in hard_dir :
            with codecs.open(hardware_dir + file, 'r') as hard_file:
                for line in hard_file:
                    doc = nlp(line)
                    for tok in doc : 
                        name = tok.text.lower()
                        if name not in feat_counts:
                            feat_counts[name] = 0
                        feat_counts[name] += 1
                nr_hard += 1
        

        # remove all features that occur less than 5 (threshold) times
        to_remove = []
        for key, value in feat_counts.items():
            if value < 5:
                to_remove.append(key)
        for key in to_remove:
            del feat_counts[key]
        

        # map feature to index
        if path == "20news-bydate-train" :
            feat_dict = {}
            i = 0
            for key in feat_counts.keys():
                feat_dict[key] = i
                i += 1
        else :
            print ("-------With threshold-------")
            feat_dict = self.train_feat_dict
        
        nr_docs = nr_rec + nr_hard
        nr_feat = len(feat_counts) 
        X = np.zeros((nr_docs, len(feat_dict)), dtype=float)
        y = np.vstack((np.zeros([nr_rec,1], dtype=int), np.ones([nr_hard,1], dtype=int)))
        nr_rec = 0
        rec_dir = os.listdir(rec_autos_dir)
        for file in rec_dir :
            with codecs.open(rec_autos_dir + file, 'r') as rec_file:
                for line in rec_file:
                    doc = nlp(line)
                    for tok in doc : 
                        name = tok.text.lower()
                        if name in feat_dict:
                            X[nr_rec,feat_dict[name]] += 1
                nr_rec += 1
        nr_hard = 0
        hard_dir = os.listdir(hardware_dir)        
        for file in hard_dir :
            with codecs.open(hardware_dir + file, 'r') as hard_file:
                for line in hard_file:
                    doc = nlp(line)
                    for tok in doc : 
                        name = tok.text.lower()
                        if name in feat_dict:
                            X[nr_rec+nr_hard,feat_dict[name]] += 1
                nr_hard += 1
        # shuffle the order, mix positive and negative examples
        new_order = np.arange(nr_docs)
        np.random.seed(0) # set seed
        np.random.shuffle(new_order)
        X = X[new_order,:]
        y = y[new_order,:]
    
        return X, y, feat_dict, feat_counts
    






        
        








