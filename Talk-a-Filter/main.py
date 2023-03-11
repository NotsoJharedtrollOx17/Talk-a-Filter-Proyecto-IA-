#DESARROLLADO EN python 3.9.8 64-bit
import basic_api_layout as REDDIT_API

def main():
    print("Hi from main!!!")
    print()
    print("Username: ")
    REDDIT_API.diplayUsername()
    print("Top subReddits:")
    REDDIT_API.displayPopularSubReddits()
    print("Hottests posts!:")
    REDDIT_API.displayHotPosts()

# application entry point
if __name__ == '__main__':
    main()