"""
Discord bot:
    - write tweets
    - detect human clasification
    - write clasified tweets
"""
import os
from discord.ext import commands

from yes_i_hate_it.config import DISCORD_TOKEN


def load_env() -> str:
    """Load discord token from environment variables"""
    return os.getenv(DISCORD_TOKEN)


class MyBot(commands.Bot):
    """Class of discord bot"""
    # engine = create_engine('sqlite://tweets.sqlite')


    def __init__(self, command_prefix, self_bot):
        """Constructor"""
        commands.Bot.__init__(self, command_prefix=command_prefix, self_bot=self_bot)
        self.add_commands()

    async def on_ready(self):
        """Execute when bot is succesfully connected"""
        print(f'{self.user.name} has connected to Discord!')

    async def on_reaction_add(self, reaction, user):
        """Execute when bot is succesfully connected"""
        if user.id == self.user.id:
            return

        messages = await reaction.message.channel.history(limit=20).flatten()
        print(messages[0])
        print('------------------------------------')
        print(messages[0].reactions)

        print(user.name)

        print(reaction.message.content)
        print(reaction)

        if str(reaction.emoji) == "✅":
            _data = {'text': reaction.message.content, 'type': 'yes'}
        elif str(reaction.emoji) == "❌":
            _data = {'text': reaction.message.content, 'type': 'no'}
        else:
            print('not emoji recongnized')
            return

    def add_commands(self):
        """add commands to bot"""
        @self.command(name="list", pass_context=True)
        # pylint: disable = redefined-builtin
        async def list(ctx):
            """Start sending data"""
            name = f'lobby-{ctx.author.name}'
            print(ctx.author.name)
            print(ctx.message.content)
            channel_id = [g.id for g in ctx.message.guild.channels if g.name == name.lower()]
            print(channel_id)
            # await ctx.channel.send("puta")
            # send 10 tweets
            msg = await self.get_channel(channel_id[0]).send("puta")
            await msg.add_reaction('✅')
            await msg.add_reaction('❌')

        @self.command(name="addme", pass_context=True)
        async def add_me(ctx):
            """Create channel for user and check it is not already created"""
            guild = ctx.message.guild
            name = f'lobby-{ctx.author.name}'
            group = [g.id for g in ctx.message.guild.channels if g.name == name.lower()]
            if [True for g in guild.channels if g.name == name.lower()]:
                await ctx.channel.send("channel already created")
            else:
                await guild.create_text_channel(name, category=group[0])

        def correct(msgs):
            corr = True
            for msg in msgs:
                corr = corr and (len([react.count for react in msg.reactions]) == 3)
            return corr

        def clasify_data(msgs):
            for _msg in msgs:
                pass

        @self.command(name="c", pass_context=True)
        async def confirm(ctx):
            """Delete all messages in users channel"""
            messages = await ctx.channel.history(limit=20).flatten()
            messages = [msg for msg in messages if '>' != msg.content[0]]
            if correct(messages):
                clasify_data(messages)
                await ctx.channel.purge()


abot = MyBot(command_prefix=">", self_bot=False)
abot.run(load_env())
