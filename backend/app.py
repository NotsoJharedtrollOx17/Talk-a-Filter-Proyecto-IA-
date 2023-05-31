from flask import Flask, jsonify
from flask_cors import CORS
from gpt4free import you
# * CONSIDERAR: https://github.com/xtekky/gpt4free/blob/main/quora/README.md

import praw
import spacy
import csv

SPACY_LANGUAGE_MODEL = "en_core_web_sm"
DATASET = "./data/data-files/csv/bad-words.csv"
OPENAI_WORKAROUND_API_MODEL = "gpt-3.5-turbo"
CONTENT_LIMIT = 5

#Flask app entry point
app = Flask(__name__)
CORS(app)

# Load spacy model and custom foul language database
nlp = spacy.load(SPACY_LANGUAGE_MODEL)
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
# TODO arreglar filtrado y entrenar modelo personalizado para detectar palabras ofensivas y analisis de sentimiento del mensaje para reddit
''''
* https://huggingface.co/docs/transformers/main_classes/pipelines
* https://towardsdatascience.com/teach-an-ai-model-to-write-like-shakespeare-for-free-a9e6a307139
* https://neptune.ai/blog/how-to-use-google-colab-for-deep-learning-complete-tutorial
* https://huggingface.co/blog/sentiment-analysis-python
* https://towardsdatascience.com/scraping-reddit-data-1c0af3040768
* https://www.geeksforgeeks.org/scraping-reddit-using-python/
'''
def filter_text(text):
    doc = nlp(text)
    has_offensive_text = False

    for token in doc:
        if token.text.lower() in foul_words:
            has_offensive_text = True
            break

    if has_offensive_text:
       censored_text = familyfriendly_text(text) 
       return censored_text

    return text

def familyfriendly_text(text):
    prompt = f"Please rewrite the following text to not include any foul language:\n\n{text}\n\nCensored text:"
    response = you.Completion.create(
        prompt = prompt
    )
    censored_text = response.text
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
        if not post.over_18:  # Check if the post is marked as NSFW
            posts.append(filter_post(post))
    
    return jsonify(posts)

if __name__ == '__main__':
    app.run()
