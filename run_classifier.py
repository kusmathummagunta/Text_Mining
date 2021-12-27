
from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes

if __name__ == '__main__':
    dataset = SentimentCorpus()
    nb = MultinomialNaiveBayes()
    
    params = nb.train(dataset.train_X, dataset.train_y)
    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)

    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    
    precision = nb.precision(predict_test, dataset.test_y)
    recall = nb.recall(predict_test, dataset.test_y)
    fscore = nb.f_score(precision,recall)
    
    
    print ("The no.of documents =",dataset.train_nr_docs,"No.of Unique Words =",len(dataset.train_feat_dict))   
    print("Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test))
    print ("F-Score",fscore)
    

