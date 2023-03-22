from flask import Flask, jsonify
from flask_cors import CORS


import praw
import spacy
import csv

LANGUAGE_MODEL = "en_core_web_sm"
DATASET = "bad-words.csv"
CONTENT_LIMIT = 5

#Flask app entry point
app = Flask(__name__)
CORS(app)

# Load spacy model and custom foul language database
nlp = spacy.load(LANGUAGE_MODEL)
foul_words = []
with open(DATASET, "r") as f:
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
def filter_text(text):
    doc = nlp(text)
    censored_text = ""
    for token in doc:
        if token.text.lower() in foul_words:
            censored_text += "*"*len(token.text) + token.whitespace_
        else:
            censored_text += token.text + token.whitespace_
    return censored_text

def filter_post(post):
    post_dict = {}
    post_dict['title'] = filter_text(post.title)
    post_dict['author'] = post.author.name if post.author else '[deleted]'
    post_dict['score'] = post.score
    post_dict['url'] = post.url
    post_dict['text'] = filter_text(post.selftext)
    post_dict['comments'] = []
    for comment in post.comments[:CONTENT_LIMIT]:
        if not comment.author:
            author_name = '[deleted]'
        else:
            author_name = comment.author.name
        comment_dict = {
            'author': author_name,
            'score': comment.score,
            'text': filter_text(comment.body)
        }
        post_dict['comments'].append(comment_dict)
    return post_dict

@app.route('/posts/<subreddit>', )
def get_posts(subreddit):
    posts = []
    subreddit_obj = reddit.subreddit(subreddit)

    for post in subreddit_obj.hot(limit=CONTENT_LIMIT):
        posts.append(filter_post(post))
    
    return jsonify(posts)

if __name__ == '__main__':
    app.run()
