import praw
import spacy
import csv

# Load spacy model and custom foul language database
nlp = spacy.load("en_core_web_sm")
foul_words = []
with open("bad-words.csv", "r") as f:
    reader = csv.reader(f, delimiter="\n")
    for row in reader:
        foul_words.append(row[0])

# Set up Reddit API
reddit = praw.Reddit(
    client_id = 'UpgNKOmI2LmyAUv0jrNnpQ',
    client_secret = '6UUvZHlDy2XgDy2IINR5NwG6wF0ubw',
    username = 'talkafilterUSER1', # talkafilterUSER1 OR talkafilterUSER2
    password = 'Expereince for AI', #NO CAMBIAR
    user_agent = 'Talk-a-Filter Personal Script' #nombre de la app registrada dentro del usuario de Reddit
)

# Define function to censor text based on custom foul language database
def censor_text(text):
    doc = nlp(text)
    censored_text = ""
    for token in doc:
        if token.text.lower() in foul_words:
            censored_text += "*"*len(token.text) + token.whitespace_
        else:
            censored_text += token.text + token.whitespace_
    return censored_text

# Get subreddit posts and censor comments

def main():
    subreddit = reddit.subreddit("wallstreetbets")
    for post in subreddit.hot(limit=5):

        print("-------------------------------")
        print("\tP O S T\t")
        print("Title:", censor_text(post.title))
        try:
            print("Author:", censor_text(post.author.name))
        except AttributeError:
            print("Author: [deleted]")
        print("Score:", post.score)
        print("URL:", post.url)
        print("Text:", censor_text(post.selftext))
        print("-------------------------------\n")
        
        print("-------------------------------")
        post.comments.replace_more(limit=0)
        print("\tC O M M E N T S\t")
        for comment in post.comments.list()[:10]:
            try:
                print("Author:", censor_text(comment.author.name))
            except AttributeError:
                print("Author: [deleted]")
            print("Score:", comment.score)
            print("Text:", censor_text(comment.body))
            print("-------------------------------\n")

if __name__ == '__main__':
    main()