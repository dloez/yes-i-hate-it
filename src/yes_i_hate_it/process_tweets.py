"""Download, store tweets and generate keyword from tweets"""
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Integer, Boolean
from sqlalchemy import create_engine

from yes_i_hate_it.config import TWEETS_DB_PATH
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
    is_football = Column(Boolean, default=False)


class User(BASE):
    """User base clase to interact with database"""
    __tablename__ = 'users'

    twitter_user_name = Column(String, primary_key=True)
    tweets_amount = Column(Integer)


def clean_users(users, db_session):
    """Remove users from list which already are at the database"""
    for user, amount in list(users):
        db_user = db_session.query(User).get(user)
        if db_user:
            users.remove((user, amount))
        else:
            db_session.add(User(twitter_user_name=user, tweets_amount=amount))
    db_session.commit()
    return users


def gather_tweets(users, db_session):
    """Gather tweets from users"""
    for user, amount in users:
        print(f"Getting {amount} tweets from {user}")

        pagination = ''
        for _ in range(0, amount, 100):
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
    if not TWEETS_DB_PATH.parents[0].exists():
        TWEETS_DB_PATH.parents[0].mkdir(exist_ok=True)

    engine = create_engine(f'sqlite:///{str(TWEETS_DB_PATH)}')
    BASE.metadata.create_all(engine)
    session_maker = sessionmaker(bind=engine)
    users = clean_users(GATHER_TWEETS_FROM_USERS, session_maker())
    gather_tweets(users, session_maker())


if __name__ == '__main__':
    main()
