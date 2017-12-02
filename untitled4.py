from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import time
import json
import get_sentiment as s
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style



#consumer key, consumer secret, access token, access secret.
ckey= "XYZ"
csecret= "XYZ"
atoken= "XYZ"
asecret= "XYZ"

class listener(StreamListener):
    def on_data(self, data):
        all_data = json.loads(data)
        
        tweet = all_data["text"]
        sentiment_value,confidence = s.sentiment(tweet)
        if confidence * 100  >= 80:
            output = open("Natural langauge processing/twitter-obama-out.txt","a")
            output.write(sentiment_value)
            output.write("\n")

        return True

    def on_error(self, status):
        print (status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["happy"])





style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    graph_data = open('Natural langauge processing/twitter-obama-out.txt','r').read()
    lines = graph_data.split('\n')
    xar = []
    yar = []
    x = 0
    y = 0
    
    for l in lines:
        x += 1
        if "pos" in l:
            y +=1
        elif "neg" in l:
            y -= 1
        xar.append(x)
        yar.append(y)
    
    ax1.clear()
    ax1.plot(xar, yar)
    
ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()