import os, logging, discord
from discord import app_commands
from src.config import DISCORD_BOT_TOKEN, GUILD_ID, STATUS_CHANNEL_ID

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
LOG_PATH = "logs/training.log"

class StatusBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        guild = discord.Object(id=GUILD_ID)                # <-- use GUILD_ID here
        self.tree.copy_global_to(guild=guild)
        await self.tree.sync(guild=guild)
        logger.info("Synced slash commands to guild %s", GUILD_ID)

    async def on_ready(self):
        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")

bot = StatusBot()

@bot.tree.command(
    name="status",
    description="Get the latest training log line",
    guild=discord.Object(id=GUILD_ID),                   # <-- use GUILD_ID here
)
async def status(interaction: discord.Interaction):
    # read the last log line
    try:
        with open(LOG_PATH, "rb") as f:
            f.seek(-1024, os.SEEK_END)
            data = f.read().decode("utf-8", errors="ignore")
        last = data.strip().splitlines()[-1]
    except Exception as e:
        last = f"⚠️ Error reading log: {e}"

    # reply in the same channel (interaction.channel is fine)
    await interaction.response.send_message(
        f"⏱ Latest log:\n```{last}```",
        ephemeral=False
    )

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN or not GUILD_ID:
        logger.error("DISCORD_BOT_TOKEN or GUILD_ID not configured!")
        exit(1)
    bot.run(DISCORD_BOT_TOKEN)

