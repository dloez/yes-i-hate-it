"""
Discord bot:
    - write tweets
    - detect human clasification
    - write clasified tweets
"""
import os
import logging
from discord.ext import commands
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

from yes_i_hate_it.config import DISCORD_TOKEN, CATEGORY_ID
from yes_i_hate_it.config import TWEETS_DB_PATH
from yes_i_hate_it.process_tweets import Tweet

def load_tkn() -> str:
    """Load discord token from environment variables"""
    return os.getenv(DISCORD_TOKEN)


class MyBot(commands.Bot):
    """Class of discord bot"""
    engine = create_engine(f'sqlite:///{str(TWEETS_DB_PATH)}')
    session_maker = sessionmaker(bind=engine)

    def __init__(self, command_prefix, self_bot):
        """Constructor"""
        commands.Bot.__init__(self, command_prefix=command_prefix, self_bot=self_bot)
        self.add_commands()

    # pylint:disable = logging-not-lazy
    async def on_ready(self):
        """Execute when bot is succesfully connected"""
        print(f'{self.user.name} has connected to Discord!')
        logging.info(self.user.name + ' has connected to Discord!')
        guild = self.guilds[1].text_channels
        print(guild)
        channels = []
        session = self.session_maker()

        for channel in guild:
            name = channel.name.split('-')
            if name[0] != 'lobby':
                continue
            if len(name) == 1:
                continue

            msgs = await channel.history(limit=100).flatten()
            if not msgs:
                print('None')
                continue
            logging.info('found msgs in lobby-' + name[1])
            for msg in msgs:
                splitted_msg = msg.content.split()
                tweet = session.query(Tweet).get(int(splitted_msg[1]))
                tweet.requested = False
                session.add(tweet)

            session.commit()
            await channel.purge()
            channels.append(channel)
            logging.info('Unrequested and deleted messages in lobby-' + name[1])


    def add_commands(self):
        """add commands to bot"""
        def read_10():
            """Read 10 tweets from data base"""
            logging.info('giving 10 new tweets')
            session = self.session_maker()
            # pylint:disable = singleton-comparison
            tweets = session.query(Tweet).filter(Tweet.requested==False).limit(10).all()
            messages=[]
            for tweet in tweets:
                tweet.requested = True
                messages.append(f'```ID: {tweet.tweet_id} \n{tweet.text}```')
            session.add_all(tweets)
            session.commit()
            return messages

        @self.command(name="list", pass_context=True)
        async def list_tweets(ctx):
            """Start sending data"""
            logging.info(ctx.author.name + 'requested list')
            name = f'lobby-{ctx.author.name}'
            channel_id = [g.id for g in ctx.message.guild.channels if g.name == name.lower()]
            msgs = read_10()
            for msg in msgs:
                msg_to_send = await self.get_channel(channel_id[0]).send(msg)
                await msg_to_send.add_reaction('✅')
                await msg_to_send.add_reaction('❌')

        @self.command(name="addme", pass_context=True)
        async def add_me(ctx):
            """Create channel for user and check it is not already created"""
            logging.info(ctx.author.name + 'rquested add me')
            guild = ctx.message.guild
            name = f'lobby-{ctx.author.name}'
            group = [g for g in ctx.message.guild.channels if g.id == CATEGORY_ID]
            if [True for g in guild.channels if g.name == name.lower()]:
                await ctx.channel.send("channel already created")
            else:
                await guild.create_text_channel(name, category=group[0])
                await list_tweets(ctx)

        def correct(msgs):
            """Check if all messages are correctly labeled"""
            corr = True
            for msg in msgs:
                corr = corr and (sum([react.count for react in msg.reactions]) == 3)
            logging.info('messages checked as' + str(corr))
            return corr

        def clasify_data(msgs):
            """Read lables and clasify tweets in data base"""
            session = self.session_maker()
            for msg in msgs:
                splitted_msg = msg.content.split()
                tweet = session.query(Tweet).get(int(splitted_msg[1]))
                if str(msg.reactions[0].emoji) == ('✅') and msg.reactions[0].count == 2:
                    tweet.is_football = True
                    session.add(tweet)
                    logging.info('clasified ' + str(tweet.tweet_id) + ' as football')

            session.commit()

        @self.command(name="c", pass_context=True)
        async def confirm(ctx):
            """Delete all messages in users channel"""
            logging.info(ctx.author.name + 'requeste confirmation')
            messages = await ctx.channel.history(limit=100).flatten()
            messages = [msg for msg in messages if '>' != msg.content[0]]
            if correct(messages):
                clasify_data(messages)
                await ctx.channel.purge()
            await list_tweets(ctx)

        @self.command(name="clean", pass_context=True)
        async def clean(ctx):
            """Delete all messages in users channel"""
            await ctx.channel.purge()


def main():
    """Main function"""
    discord_bot = MyBot(command_prefix=">", self_bot=False)
    discord_bot.run(load_tkn())


if __name__ == '__main__':
    logging.basicConfig(filename='process.log', format='%(asctime)s %(message)s', level=logging.INFO, force=True)
    main()
