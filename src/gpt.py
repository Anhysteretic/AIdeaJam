import os, re, asyncio, time, json
from collections import deque, defaultdict
from pathlib import Path
from typing import List, Optional

import discord
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI  # async client

# --- env ---
env_path = Path(__file__).resolve().parent.parent / '.env'
load_dotenv(dotenv_path=env_path, verbose=True)
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
MODEL = 'gpt-4o-mini'  # use a model that supports Structured Outputs for .parse

# --- discord client + intents ---
intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

# --- openai client (async) ---
oai = AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=20.0)
LLM_SEMA = asyncio.Semaphore(2)  # limit concurrent LLM calls

# --- rolling context per channel ---
RECENT = defaultdict(lambda: deque(maxlen=30))   # last 30 msgs per channel
LAST_INTERJECT_TS = defaultdict(lambda: 0.0)
USER_COOLDOWN = defaultdict(lambda: 0.0)
CHANNEL_COOLDOWN_SEC = 0
USER_COOLDOWN_SEC = 0
MAX_HISTORY_BACKFILL = 50

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
    if content.endswith("?"):
        return True
    if HELP_REGEX.search(content):
        return True
    joined = " ".join(m["text"] for m in list(RECENT[message.channel.id])[-8:])
    if re.search(r"(confus|not sure|don.?t understand|incorrect)", joined, re.I):
        return True
    return False

SYSTEM_INSTR = (
    "You are a teaching aide in a student Discord. Keep replies short, friendly, and actionable. "
    "Do not over-explain. Offer a hint, link to a concept, or a next step. If the question is already "
    "answered, or unclear, do NOT interject. NEVER leak private data or make up course policies."
)

def _as_text(msg: discord.Message) -> str:
    content = (msg.content or "").strip()
    if not content and msg.attachments:
        names = ", ".join(a.filename for a in msg.attachments[:3])
        content = f"[attachments: {names}]"
    return content

async def load_recent_for_channel(ch: discord.TextChannel):
    try:
        perms = ch.permissions_for(ch.guild.me)
        if not (perms.read_messages and perms.read_message_history):
            return
        async for msg in ch.history(limit=MAX_HISTORY_BACKFILL, oldest_first=True):
            RECENT[ch.id].append({
                "author": str(getattr(msg.author, "display_name", msg.author.name)),
                "text": _as_text(msg),
                "bot": bool(getattr(msg.author, "bot", False)),
            })
    except Exception as e:
        print(f"[prime] failed for #{getattr(ch, 'name', ch.id)}: {e}")

async def call_llm_structured(context_lines: List[str]) -> Optional[Decision]:
    # First try: Structured Outputs parse
    try:
        async with LLM_SEMA:
            completion = await asyncio.wait_for(
                oai.beta.chat.completions.parse(
                    model=MODEL,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_INSTR},
                        {"role": "user", "content":
                            "Below are recent Discord messages. Decide if it's helpful to chime in now.\n"
                            "If you speak, be concise (<= 4 sentences). If someone asked for code, provide a small snippet.\n"
                            "Context:\n" + "\n".join(context_lines)
                        }
                    ],
                    response_format=Decision,
                ),
                timeout=18.0,
            )
        parsed = completion.choices[0].message.parsed
        if parsed is not None:
            return parsed
    except asyncio.TimeoutError:
        print("[LLM] parse timed out")
    except Exception as e:
        print(f"[LLM] parse failed: {e}")

    # Fallback: JSON mode + Pydantic validation
    try:
        async with LLM_SEMA:
            completion = await asyncio.wait_for(
                oai.chat.completions.create(
                    model=MODEL,
                    temperature=0.2,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": (
                            SYSTEM_INSTR +
                            " Return ONLY a JSON object with keys: "
                            "should_interject (bool), confidence (0..1), reason (string), "
                            "reply (string|nullable), tags (string list)."
                        )},
                        {"role": "user", "content":
                            "Below are recent Discord messages. Decide if it's helpful to chime in now.\n"
                            "If you speak, be concise (<= 4 sentences). If someone asked for code, provide a small snippet.\n"
                            "Context:\n" + "\n".join(context_lines)
                        }
                    ],
                ),
                timeout=18.0,
            )
        raw = completion.choices[0].message.content
        return Decision.model_validate(json.loads(raw))
    except asyncio.TimeoutError:
        print("[LLM] json fallback timed out")
    except Exception as e:
        print(f"[LLM] json fallback failed: {e}")
    return None

async def maybe_interject(message: discord.Message):
    if not gate_should_consider(message):
        return
    if not allowed_to_attempt(message.channel.id, message.author.id):
        return

    context_lines = [
        f'- {m["author"]}: {m["text"]}'
        for m in list(RECENT[message.channel.id])[-18:]
    ]

    result = await call_llm_structured(context_lines)
    if not result:
        return

    if result.should_interject and result.reply:
        reply = result.reply.strip()
        if 0 < len(reply) <= 1200:
            await message.channel.send(reply)
            LAST_INTERJECT_TS[message.channel.id] = time.time()
            USER_COOLDOWN[message.author.id] = time.time()

@client.event
async def on_ready():
    print(f"Logged in as {client.user}")
    # Prime recent context for all visible text channels
    for guild in client.guilds:
        for ch in guild.text_channels:
            await load_recent_for_channel(ch)
    print("[prime] backfill complete")

@client.event
async def on_message(message: discord.Message):
    # store rolling context
    RECENT[message.channel.id].append({
        "author": str(message.author.display_name),
        "text": message.content or "",
        "bot": message.author.bot
    })

    if message.author == client.user:
        return

    if message.content.strip().endswith("?"):
        asyncio.create_task(delayed_attempt(message, delay=25))
    else:
        asyncio.create_task(maybe_interject(message))

async def delayed_attempt(message: discord.Message, delay: int = 25):
    await asyncio.sleep(delay)
    recent_text = " ".join(m["text"] for m in list(RECENT[message.channel.id])[-5:])
    if re.search(r"(nvm|nevermind|figured it out|got it|thanks)", recent_text, re.I):
        return
    await maybe_interject(message)

if __name__ == "__main__":
    client.run(DISCORD_TOKEN)
