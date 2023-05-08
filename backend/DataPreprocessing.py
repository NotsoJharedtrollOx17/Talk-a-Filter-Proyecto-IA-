import praw
import spacy
import csv
import re

SPACY_LANGUAGE_MODEL = "en_core_web_sm"
WALLSTREETBETS = "wallstreetbets"
DATASET = "./data/bad-words.csv"
GENERATED_DATASET = "./data/scraped-content.csv"
CONTENT_LIMIT = 5

# Set up Reddit API
reddit = praw.Reddit(
    client_id = 'UpgNKOmI2LmyAUv0jrNnpQ',
    client_secret = '6UUvZHlDy2XgDy2IINR5NwG6wF0ubw',
    username = 'talkafilterUSER2', # talkafilterUSER1 OR talkafilterUSER2
    password = 'Expereince for AI', #NO CAMBIAR
    user_agent = 'Talk-a-Filter Personal Script' #nombre de la app registrada dentro del usuario de Reddit
)
subreddit = reddit.subreddit(WALLSTREETBETS)

# Load spacy model and custom foul language database
language_model_engine = spacy.load(SPACY_LANGUAGE_MODEL)

text_dataframe = ''
foul_words_dataset = []
detected_foul_words = []

def load_offensive_words_dataset():
    with open(DATASET, "r") as f:
        reader = csv.reader(f, delimiter="\n")
        for row in reader:
            foul_words_dataset.append(row[0])
        f.close()

def write_custom_instruction(element):
    detected_foul_words = element
    instruction_template = f'The following text was flagged because it has the following offensive words: ['

    for word in detected_foul_words:
        instruction_template += f'{word}, ' 

    instruction_template = instruction_template + '], rewrite it in such a way that it does not include any of the provided words or any offensive context: '
    
    return instruction_template

def has_offensive_text(text):
    tokenized_text = language_model_engine(text)
    has_offensive_text = False

    # ? it has offensive text
    for token in tokenized_text:
        if token.text.lower() in foul_words_dataset:
            has_offensive_text = True
            detected_foul_words.append(token.text.lower())
            continue

    return has_offensive_text

def display_scraped_items(items):
    print("\tScraped text content:")
    for index, item in enumerate(items):
        print(f'No. {index}:\n Instruction: {item[0]}\n Text to Rewrite: {item[1]}\n')

def get_posts():
    print("get_posts")
    print("\tBEGIN\t\n")
    posts = subreddit.top(time_filter='year', limit=CONTENT_LIMIT)
    dataset = []

    # * iterating every submission on the TOAT of WSB
    for submission in posts:
        post_dataset_entry = {}
        comment_dataset_entry = {}
        
        # ? is text-based?
        if not submission.is_self:
                    # * checks for text in the comments...
            for comment in submission.comments[:CONTENT_LIMIT]:
                if isinstance(comment, praw.models.MoreComments):
                  continue
                if comment.body != '[deleted]' or comment.body != '[removed]': # exclude deleted or removed comments
                    if comment.body_html is not None:
                        comment.body_html = ''

                    if not re.match(r'^https?:\/\/.*[\r\n]*', comment.body): # exclude comments with links
                        if has_offensive_text(comment.body):
                            curated_comment = comment.body.replace('\n', ' ')
                            comment_dataset_entry = [write_custom_instruction(detected_foul_words), curated_comment]
                            dataset.append(comment_dataset_entry)
                            detected_foul_words.clear()

            continue

            # ? if the content does not have any URLs, or wasn't deleted/removed...
        if submission.selftext != '[deleted]' or submission.selftext != '[removed]': # exclude deleted or removed posts
            if submission.selftext_html is not None:
                submission.selfttext_html = ''

            if not re.match(r'^https?:\/\/.*[\r\n]*', submission.selftext): # exclude comments with links
                    if has_offensive_text(submission.selftext):
                        post_dataset_entry = [write_custom_instruction(detected_foul_words), submission.selftext]
                        dataset.append(post_dataset_entry)
                        detected_foul_words.clear()

            # * checks for text in the comments...
            for comment in submission.comments[:CONTENT_LIMIT]:
                if isinstance(comment, praw.models.MoreComments):
                  continue

                if comment.body != '[deleted]' or comment.body != '[removed]': # exclude deleted or removed comments
                    if comment.body_html is not None:
                        comment.body_html = ''

                    if not re.match(r'^https?:\/\/.*[\r\n]*', comment.body): # exclude comments with links
                        if has_offensive_text(comment.body):
                            curated_comment = comment.body.replace('\n', ' ')
                            comment_dataset_entry = [write_custom_instruction(detected_foul_words), curated_comment]
                            dataset.append(comment_dataset_entry)
                            detected_foul_words.clear()

            post_dataset_entry.clear()
            comment_dataset_entry.clear()    

    display_scraped_items(dataset)

    print(F"\nSaving data to {GENERATED_DATASET}\n")
    with open(GENERATED_DATASET, mode='w', newline='') as data_file:
        writer = csv.writer(data_file)
        writer.writerows(dataset)
        data_file.close()
    print(f"\nDataset generated to {GENERATED_DATASET}\n")

    print("\tEND\t")

def main():
    load_offensive_words_dataset()
    print("WSB Comment Scraper\n")
    get_posts()

if __name__ == '__main__':
    main()