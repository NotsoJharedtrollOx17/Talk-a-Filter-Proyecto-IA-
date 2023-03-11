#libreria para el funcionamiento del script
import praw
    # como instalar: pip install praw

REDDIT_API_INSTANCE = praw.Reddit(
    client_id = 'UpgNKOmI2LmyAUv0jrNnpQ',
    client_secret = '6UUvZHlDy2XgDy2IINR5NwG6wF0ubw',
    username = 'talkafilterUSER1', # talkafilterUSER1 OR talkafilterUSER2
    password = 'Expereince for AI', #NO CAMBIAR
    user_agent = 'Talk-a-Filter Personal Script' #nombre de la app registrada dentro del usuario de Reddit
)

CONTENT_LIMIT = 5

def getUsername():
    username = REDDIT_API_INSTANCE.user.me()
    return username

def getPopularSubReddits():
    subreddits = REDDIT_API_INSTANCE.subreddits.popular(limit=CONTENT_LIMIT)
    return subreddits

def getHotPosts():
    hot_posts = REDDIT_API_INSTANCE.front.hot(limit=CONTENT_LIMIT)
    return hot_posts

def diplayUsername():
    print(getUsername())
    print()

def displayPopularSubReddits():
    for id, subreddit in enumerate(getPopularSubReddits()):
        print(id+1, ": ", subreddit.display_name)

    print()

def displayHotPosts():
    for id, post in enumerate(getHotPosts()):
        print(id+1, ": ", post.title, "Upvotes: ", post.ups, "Downvotes", post.downs)

    print()