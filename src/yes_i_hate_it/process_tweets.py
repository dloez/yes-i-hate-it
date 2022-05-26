"""Download, store tweets and generate keyword from tweets"""
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Boolean
from sqlalchemy import create_engine

from yes_i_hate_it.config import GATHER_TWEETS_FROM_USERS
from yes_i_hate_it.config import BASE

from yes_i_hate_it.main import get_tweets


# pylint: disable = too-few-public-methods
class Tweet(BASE):
    """Tweet base clase to interact with database"""
    __tablename__ = 'tweets'

    tweet_id = Column(Integer, primary_key=True)
    text = Column(String)
    requested = Column(Boolean, default=False)


class User(BASE):
    """User base clase to interact with database"""
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True)
    twitter_user_name = Column(String)


def gather_tweets(users, db_session):
    """Gather tweets from users"""
    for user in users:
        pagination = ''
        for _ in range(30):
            tweets = get_tweets(
                user_name=user,
                max_results=100,
                pagination=pagination
            )

            pagination = tweets.meta['next_token']
            tweets = tweets.data or []
            if not tweets:
                break

            tweet_storage = []
            for tweet in tweets:
                tweet_storage.append(Tweet(
                    tweet_id=tweet.id,
                    text=tweet.text
                ))
            db_session.add_all(tweet_storage)
            db_session.commit()


def main():
    """Main function"""
    engine = create_engine('sqlite:///tweets.sqlite')
    BASE.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)

    gather_tweets(users=GATHER_TWEETS_FROM_USERS, db_session=session_maker())


if __name__ == '__main__':
    main()
