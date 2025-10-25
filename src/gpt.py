import os, re, asyncio, time
from collections import deque, defaultdict
import discord
from pydantic import BaseModel, Field, conlist
from typing import List, Optional
from openai import OpenAI
from pathlib import Path  # Recommended for path manipulation
from dotenv import load_dotenv


env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path, verbose=True)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-4o-mini'  # supports Structured Outputs per OpenAI docs. :contentReference[oaicite:1]{index=1}


intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)
oai = OpenAI(api_key=OPENAI_API_KEY)


# --- rolling context per channel ---
# <<< MODIFICATION 1: Changed maxlen from 30 to 50 >>>
RECENT = defaultdict(lambda: deque(maxlen=50))  # keep last 50 messages per channel
LAST_INTERJECT_TS = defaultdict(lambda: 0.0)    # channel cooldown
USER_COOLDOWN = defaultdict(lambda: 0.0)        # user cooldown
CHANNEL_COOLDOWN_SEC = 0
USER_COOLDOWN_SEC = 0


HELP_REGEX = re.compile(r"(anyone know|how (do|to) i|stuck|help|can someone|does anyone|what.*mean)", re.I)


class Decision(BaseModel):
   should_interject: bool = Field(..., description="Whether to speak now.")
   confidence: float = Field(..., ge=0, le=1)
   reason: str
   reply: Optional[str] = Field(None, description="Bot's actual message to send if interjecting.")
   tags: List[str] = Field(default_factory=list)


def allowed_to_attempt(channel_id: int, author_id: int) -> bool:
   now = time.time()
   if now - LAST_INTERJECT_TS[channel_id] < CHANNEL_COOLDOWN_SEC:
       return False
   if now - USER_COOLDOWN[author_id] < USER_COOLDOWN_SEC:
       return False
   return True


def gate_should_consider(message: discord.Message) -> bool:
   if message.author.bot:
       return False
   content = (message.content or "").strip()
   if len(content) < 2:
       return False
   if content.startswith("```") and content.endswith("```"):
       return False
   # heuristic triggers
   if content.endswith("?"):
       return True
   if HELP_REGEX.search(content):
       return True
   # if recent confusion markers
   joined = " ".join(m["text"] for m in list(RECENT[message.channel.id])[-8:])
   if re.search(r"(confus|not sure|don.?t understand|incorrect)", joined, re.I):
       return True
   return False


def build_chat_messages(channel_id: int) -> List[dict]:
   msgs = []
   for m in RECENT[channel_id]:
       role = "user" if not m["bot"] else "assistant"
       msgs.append({"role": role, "content": f'{m["author"]}: {m["text"]}'})
   return msgs[-18:]  # keep prompt lean


SYSTEM_INSTR = (
   "You are a teaching aide in a student Discord. Keep replies short, friendly, and actionable. "
   "Do not over-explain. Offer a hint, link to a concept, or a next step. If the question is already "
   "answered, or unclear, do NOT interject. NEVER leak private data or make up course policies."
)


async def maybe_interject(message: discord.Message):
   if not gate_should_consider(message):
       return
   if not allowed_to_attempt(message.channel.id, message.author.id):
       return


   chat_msgs = build_chat_messages(message.channel.id)


   # --- Structured output call (Pydantic parse helper) ---
   completion = oai.beta.chat.completions.parse(
       model=MODEL,
       messages=[
           {"role": "system", "content": SYSTEM_INSTR},
           {"role": "user", "content":
            "Below are recent Discord messages. Decide if it's helpful to chime in now.\n"
            "If you speak, be concise (<= 4 sentences). If someone asked for code, provide a small snippet.\n"
            "Context:\n" + "\n".join(f'- {m["author"]}: {m["text"]}' for m in RECENT[message.channel.id])}
       ],
       response_format=Decision,  # <-- Structured Outputs → returns a Decision instance
   )
   parsed = completion.choices[0].message.parsed
   if parsed is None:
       print(f'message was not parsed')
       return
   result: Decision = parsed  # typed object


   print(result)


   if result.should_interject and result.reply:
       # safety: double-check length and basic sanity
       reply = result.reply.strip()
       if 0 < len(reply) <= 1200:
           await message.channel.send(reply)
           LAST_INTERJECT_TS[message.channel.id] = time.time()
           USER_COOLDOWN[message.author.id] = time.time()


@client.event
async def on_ready():
   print(f"Logged in as {client.user}")

   # <<< MODIFICATION 2: Added history fetching on startup >>>
   print("Fetching initial message history for context...")
   for guild in client.guilds:
       for channel in guild.text_channels:
           # Make sure the bot has permissions to read history
           if channel.permissions_for(guild.me).read_message_history:
               print(f" - Populating context for #{channel.name} in {guild.name}")
               try:
                   # Fetch last 50 messages. Note: They come in newest-to-oldest order.
                   async for message in channel.history(limit=50):
                       # Use appendleft to add oldest messages first, maintaining chronological order
                       RECENT[channel.id].appendleft({
                           "author": str(message.author.display_name),
                           "text": message.content or "",
                           "bot": message.author.bot
                       })
                   print(f"   -> Done: Loaded {len(RECENT[channel.id])} messages.")
               except discord.Forbidden:
                   print(f"   -> FAILED: No permission to read history in #{channel.name}.")
               except Exception as e:
                   print(f"   -> FAILED: An error occurred in #{channel.name}: {e}")
           else:
               print(f" - Skipping #{channel.name} (no history permission)")
   print("Initial context populated.")
   # <<< END OF MODIFICATION 2 >>>


@client.event
async def on_message(message: discord.Message):
   # store rolling context
   RECENT[message.channel.id].append({
       "author": str(message.author.display_name),
       "text": message.content or "",
       "bot": message.author.bot
   })


   print(f'messageRecieved: {message.content}')
   # never reply to ourselves
   if message.author == client.user:
       return


   # simple “unanswered question” timeout: check again in 25s if the last message is a question
   if message.content.strip().endswith("?"):
       asyncio.create_task(delayed_attempt(message, delay=25))
   else:
       # otherwise try immediately if heuristic gates say so
       asyncio.create_task(maybe_interject(message))


async def delayed_attempt(message: discord.Message, delay: int = 25):
   await asyncio.sleep(delay)
   # if new messages arrived that likely answered, skip
   recent_text = " ".join(m["text"] for m in list(RECENT[message.channel.id])[-5:])
   if re.search(r"(nvm|nevermind|figured it out|got it|thanks)", recent_text, re.I):
       return
   await maybe_interject(message)


client.run(DISCORD_TOKEN)