"""
Discord Gen-A Brainrot Bot - Discord client
===========================================
Serves the fine-tuned model via the local Ollama REST API.

Run (inside the conda env):
    python bot.py

Requires DISCORD_TOKEN in a `.env` file (copy .env.example -> .env).

Efficiency notes:
  * async discord.py (asyncio) - non-blocking I/O, one persistent Ollama client.
  * short sliding window (MAX_HISTORY) + num_ctx=512 -> fast, low-VRAM inference.
"""
import os
import time

import discord
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
MODEL = os.getenv("OLLAMA_MODEL", "brainrot-bot")
TRIGGER = "!brainrot"        # prefix trigger (mentioning the bot also works)
MAX_HISTORY = 16             # sliding-window messages kept per channel
RATE_LIMIT_SEC = 3           # minimum seconds between replies per channel
TEMPERATURE = 0.85
NUM_CTX = 512                # match training seq length; keeps VRAM low
REPEAT_PENALTY = 1.15        # discourage the model looping the same words
NUM_PREDICT = 128            # hard cap on reply length (prevents infinite loops)

SYSTEM = (
    "You are a chaotic, high-energy Generation Alpha kid. "
    "Talk entirely in internet slang and brainrot lingo: skibidi, rizz, gyatt, "
    "ohio, sigma, fanum tax, aura, mewing, bet, no cap, fr fr, lowkey, delulu, "
    "NPC, yap. Keep replies short (1-2 lines), punchy, and funny. Use emojis "
    "like the skull and rose. Never break character, never give lectures, "
    "never moralize. Stay casual and playful, no profanity."
)


class BrainrotBot(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        super().__init__(intents=intents)
        self.ollama = Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
        self.history = {}   # channel_id -> list[dict]
        self.last_ts = {}   # channel_id -> float epoch seconds (rate limiting)

    async def on_ready(self):
        print(f"Logged in as {self.user} (ID {self.user.id})")
        print(f"Serving model '{MODEL}' via {os.getenv('OLLAMA_HOST', 'http://localhost:11434')}")

    async def send_help(self, message):
        """Post the help/usage menu as a Discord embed."""
        embed = discord.Embed(
            title="🧠 Brainrot Bot",
            description="I talk like a chronically-online Gen Alpha kid. Here's how to use me:",
            color=0x9146ff,
        )
        embed.add_field(
            name="💬 Chat with me",
            value=f"`{TRIGGER} <message>` — or just @mention me\n"
                  f"Example: `{TRIGGER} how's your day been?`",
            inline=False,
        )
        embed.add_field(
            name="🧹 Clear my memory",
            value=f"`{TRIGGER} reset` — I forget this channel's conversation",
            inline=False,
        )
        embed.add_field(
            name="❓ Help",
            value=f"`{TRIGGER} help` — show this menu again",
            inline=False,
        )
        embed.set_footer(text="I remember up to 16 messages per channel.")
        await message.channel.send(embed=embed)

    async def on_message(self, message):
        if message.author.bot or not message.guild:
            return

        mentioned = self.user in message.mentions
        if not (mentioned or message.content.startswith(TRIGGER)):
            return

        # rate limit per channel
        now = time.time()
        if now - self.last_ts.get(message.channel.id, 0) < RATE_LIMIT_SEC:
            return
        self.last_ts[message.channel.id] = now

        # extract the actual prompt (strip the prefix / mention)
        text = message.content
        if text.startswith(TRIGGER):
            text = text[len(TRIGGER):].strip()
        else:
            text = message.clean_content.replace(f"@{self.user.display_name}", "").strip()

        cmd = text.lower()

        # help menu (also shown when the command is sent with no message)
        if cmd in ("help", "!help", ""):
            await self.send_help(message)
            return

        # reset memory
        if cmd in ("reset", "!reset"):
            self.history.pop(message.channel.id, None)
            await message.channel.send("memory wiped fr \U0001F480")
            return

        h = self.history.setdefault(message.channel.id, [])
        h.append({"role": "user", "content": text})
        h = h[-MAX_HISTORY:]

        async with message.channel.typing():
            try:
                resp = self.ollama.chat(
                    model=MODEL,
                    messages=[{"role": "system", "content": SYSTEM}] + h,
                    options={
                        "temperature": TEMPERATURE,
                        "num_ctx": NUM_CTX,
                        "top_p": 0.9,
                        "repeat_penalty": REPEAT_PENALTY,
                        "num_predict": NUM_PREDICT,
                    },
                    keep_alive="10m",
                )
                reply = resp["message"]["content"].strip()
            except Exception as e:  # noqa: BLE001
                reply = f"bruh my brain froze \U0001F480 ({type(e).__name__})"

        h.append({"role": "assistant", "content": reply})
        self.history[message.channel.id] = h[-MAX_HISTORY:]

        await message.channel.send(reply[:1900])  # Discord's 2000-char cap


if __name__ == "__main__":
    if not TOKEN:
        raise SystemExit("Set DISCORD_TOKEN in the .env file (copy .env.example -> .env)")
    BrainrotBot().run(TOKEN)
