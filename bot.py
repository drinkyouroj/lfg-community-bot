import os
import logging
import discord
from discord.ext import commands
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Bot setup
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

class LFGBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        self.initial_extensions = [
            'cogs.rag_cog'
        ]

    async def setup_hook(self):
        # Load all extensions
        for extension in self.initial_extensions:
            try:
                await self.load_extension(extension)
                logger.info(f'Loaded extension: {extension}')
            except Exception as e:
                logger.error(f'Failed to load extension {extension}: {e}')

    async def on_ready(self):
        logger.info(f'Logged in as {self.user} (ID: {self.user.id})')
        await self.change_presence(
            activity=discord.Activity(
                type=discord.ActivityType.listening,
                name='your questions | !help'
            )
        )

# Create and run the bot
if __name__ == '__main__':
    bot = LFGBot()
    bot.run(os.getenv('DISCORD_TOKEN'))
