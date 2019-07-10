import numpy as np
import tweepy, json, re, csv
from time import gmtime, strftime, strptime
from keras.models import load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

def compute_weight():
        
    def FindURL(string): return re.findall("http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", string)
    
    def PreProcess(text):
        
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text)
        X = tokenizer.texts_to_sequences(text)
        X = np.array(sequence.pad_sequences(X, maxlen=50))
        
        return X
    
    api_key = 'noY60ockEQUfjFdz1fS3b7ZeB'
    api_secret_key = 'itE8jzxQ8BXDmh4Le61m4fHgM90X3IfK1pf7URGQmxEOuYi6si'
    access_token = '4316836098-ef4Nwfg8g49DSbFdcn7B2DpEXaMo9TBJAuqp3ya'
    access_token_secret = '4E6ARa8DKqJktJ5rE0MakJ3u4Eg47gWovYIeajOyUgghm'
    
    auth = tweepy.OAuthHandler(api_key, api_secret_key)
    auth.set_access_token(access_token, access_token_secret)
            
    auth_api = tweepy.API(auth)
    
    with open("CBC_Montreal.csv", "w", newline='') as fp:
        csv.writer(fp, dialect='excel').writerow(['ID', 'Time', 'Tweet'])    
    
    data = tweepy.API(auth).user_timeline(screen_name = "CBCMontreal", tweet_mode = 'extended')
    
    news_weight = []
    
    for tweet in data:
                
        tweet = json.loads(json.dumps(tweet._json))
        
        if strftime("%d", gmtime()) == strftime('%d', strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')):
            
            model = load_model("../outputs/model.hdf5")
    
            with open("CBC_Montreal.csv", "a+", newline='', encoding="utf-8") as fp:
                
                headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/35.0.1916.47 Safari/537.36'}
                
                if len(FindURL(tweet['full_text'])) != 0: tweet_text = tweet['full_text'].replace(FindURL(tweet['full_text'])[0], "")
                else: tweet_text = tweet['full_text']
                    
                pred = model.predict(PreProcess(tweet_text))
                prediction = sum(pred)/len(pred)
                
                index_of_maximum = np.where(prediction == np.max(prediction))
                
                if index_of_maximum[0][0] == 0: weight = 0
                elif index_of_maximum[0][0] == 1: weight = 0.95
                elif index_of_maximum[0][0] == 2: weight = 0.1
                elif index_of_maximum[0][0] == 3: weight = 0.1
                elif index_of_maximum[0][0] == 4: weight = 0.5
                elif index_of_maximum[0][0] == 5: weight = 0.1
                elif index_of_maximum[0][0] == 6: weight = 0.25
                elif index_of_maximum[0][0] == 7: weight = 0.1
                elif index_of_maximum[0][0] == 8: weight = 0.25
                elif index_of_maximum[0][0] == 9: weight = 0.1
                elif index_of_maximum[0][0] == 10: weight = 0.1
                elif index_of_maximum[0][0] == 11: weight = 0
                elif index_of_maximum[0][0] == 12: weight = 0
                elif index_of_maximum[0][0] == 13: weight = 0
                elif index_of_maximum[0][0] == 14: weight = 0
                elif index_of_maximum[0][0] == 15: weight = 0
                elif index_of_maximum[0][0] == 16: weight = 0.75
                elif index_of_maximum[0][0] == 17: weight = 0.1
                elif index_of_maximum[0][0] == 18: weight = 0.5
                elif index_of_maximum[0][0] == 19: weight = 0
                elif index_of_maximum[0][0] == 20: weight = 0
                elif index_of_maximum[0][0] == 21: weight = 0.95
                elif index_of_maximum[0][0] == 22: weight = 0.75
                elif index_of_maximum[0][0] == 23: weight = 0.95
                elif index_of_maximum[0][0] == 24: weight = 0.1
                elif index_of_maximum[0][0] == 25: weight = 0.1
                elif index_of_maximum[0][0] == 26: weight = 0
                elif index_of_maximum[0][0] == 27: weight = 0
                elif index_of_maximum[0][0] == 28: weight = 0
                elif index_of_maximum[0][0] == 29: weight = 0
                elif index_of_maximum[0][0] == 30: weight = 0
                elif index_of_maximum[0][0] == 31: weight = 0.5
                elif index_of_maximum[0][0] == 32: weight = 0.25
                elif index_of_maximum[0][0] == 33: weight = 0.25
                elif index_of_maximum[0][0] == 34: weight = 0.25
                
                csv.writer(fp).writerow([tweet['id'], 
                                         strftime('%Y-%m-%d %H:%M:%S', strptime(tweet['created_at'], '%a %b %d %H:%M:%S +0000 %Y')),
                                         tweet['full_text'],
                                         weight])
                
                news_weight.append(weight)    
    
    news_weight = sum(news_weight) / len(news_weight)
    print(news_weight)
    
    return news_weight