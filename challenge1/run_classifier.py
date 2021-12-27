
from sentiment_reader import SentimentCorpus
from multinomial_naive_bayes import MultinomialNaiveBayes

if __name__ == '__main__':
    dataset = SentimentCorpus()
    dataset_new =  SentimentCorpus(train_per=0.5, dev_per=0, test_per=0.5) 
    nb = MultinomialNaiveBayes()
    
    params = nb.train(dataset.train_X, dataset.train_y)
    params_new = nb.train(dataset_new.train_X, dataset_new.train_y)

    
    predict_train = nb.test(dataset.train_X, params)
    eval_train = nb.evaluate(predict_train, dataset.train_y)
    predict_train_new = nb.test(dataset_new.train_X, params_new)
    eval_train_new = nb.evaluate(predict_train_new, dataset_new.train_y)
    
    predict_test = nb.test(dataset.test_X, params)
    eval_test = nb.evaluate(predict_test, dataset.test_y)
    predict_test_new = nb.test(dataset_new.test_X, params_new)
    eval_test_new = nb.evaluate(predict_test_new, dataset_new.test_y)
    
    print ("---------train(80%) and test(20%)-----------")    
    print("Accuracy on training set: %f, on test set: %f" % (eval_train, eval_test))
    
    print("-----------train(50%) and test(50%)-----------")
    print("Accuracy on training set: %f, on test set: %f" % (eval_train_new, eval_test_new))



