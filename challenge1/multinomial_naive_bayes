import numpy as np
from linear_classifier import LinearClassifier


class MultinomialNaiveBayes(LinearClassifier):

    def __init__(self):
        LinearClassifier.__init__(self)
        self.trained = False
        self.likelihood = 0
        self.prior = 0
        self.smooth = True
        self.smooth_param = 1
        
    def train(self, x, y):
        # n_docs = no. of documents
        # n_words = no. of unique words    
        n_docs, n_words = x.shape
        
        # classes = a list of possible classes
        classes = np.unique(y)
        
        # n_classes = no. of classes
        n_classes = np.unique(y).shape[0]
        
        # initialization of the prior and likelihood variables
        prior = np.zeros(n_classes)
        likelihood = np.zeros((n_words,n_classes))

        # TODO: This is where you have to write your code!
        # You need to compute the values of the prior and likelihood parameters
        # and place them in the variables called "prior" and "likelihood".
        # Examples:
            # prior[0] is the prior probability of a document being of class 0
            # likelihood[4, 0] is the likelihood of the fifth(*) feature being 
            # active, given that the document is of class 0
            # (*) recall that Python starts indices at 0, so an index of 4 
            # corresponds to the fifth feature!
        
        ###########################

        negatives =0
        
        for i in y[:] :
            if i == 0:
                negatives += 1
                
        positives = y.shape[0]-negatives
                
        prior[0] = negatives /y.shape[0]
        prior[1] = positives /y.shape[0]
        
        neg_set , pos_set = [] , []
                
        k = 0
        for i in y[:]:
            if i == 0:
                neg_set.append(x[k])
            elif i == 1:
                pos_set.append(x[k])
            k += 1
           
        pos_count = np.sum(pos_set, axis=0)
        neg_count = np.sum(neg_set, axis=0)
        
        pos_total = np.sum(pos_count)
        neg_total = np.sum(neg_count)
                    
        pos_likelihood = np.zeros(len(pos_count))
        neg_likelihood = np.zeros(len(neg_count))
                        
                         
        for i in range(len(pos_count)):
            pos_likelihood[i] = ((pos_count[i])+ 1)/(pos_total + n_words )
            neg_likelihood[i] = ((neg_count[i])+ 1) /(neg_total + n_words )
            likelihood[i][0] , likelihood[i][1]  = neg_likelihood[i] , pos_likelihood[i]
                           
        
        ###########################

        params = np.zeros((n_words+1,n_classes))
        for i in range(n_classes): 
            # log probabilities
            params[0,i] = np.log(prior[i])
            with np.errstate(divide='ignore'): # ignore warnings
                params[1:,i] = np.nan_to_num(np.log(likelihood[:,i]))
        self.likelihood = likelihood
        self.prior = prior
        self.trained = True
        return params
