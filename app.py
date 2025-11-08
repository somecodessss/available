import os
import asyncio
import json
import re
import hashlib
from datetime import datetime, timezone

import aiohttp
import discord
from discord.ext import tasks
from discord import app_commands
from bs4 import BeautifulSoup
from fastapi import FastAPI
import uvicorn

# ------------------- tiny HTTP server for Render health checks -------------------
api = FastAPI()

@api.get("/health")
async def health():
    return {"ok": True}

# ------------------- config -------------------
TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["DISCORD_CHANNEL_ID"])
GUILD_ID = int(os.getenv("GUILD_ID", "0"))  # optional: speeds up command sync
STATE_FILE = os.getenv("STATE_FILE", "state.json")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))

URL_IRSAFAM = os.getenv("TIMETABLE_URL", "https://irsafam.org/ielts/timetable")
URL_TEHRAN = os.getenv("TIMETABLE_URL_2", "https://ieltstehran.com/computer-delivered-ielts-exam/")

# ------------------- discord client -------------------
intents = discord.Intents.default()
client = discord.Client(intents=intents, allowed_mentions=discord.AllowedMentions(everyone=True))
tree = app_commands.CommandTree(client)

# ------------------- utils -------------------
def normalize_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def fa_to_en_digits(s: str) -> str:
    return s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))

def cleanse_text(t: str) -> str:
    t = fa_to_en_digits(t)
    t = t.replace("٫", ":").replace("ː", ":").replace("،", ",")
    return normalize_ws(t)

STATUS_FULL_PATTERNS = (
    "تکمیل ظرفیت",
    "ظرفیت تکمیل",
    "full",
    "closed",
    "no seats",
    "sold out",
)
STATUS_OPEN_PATTERNS = (
    "ثبت نام",
    "ثبت‌نام",
    "رزرو",
    "reserve",
    "book",
    "available",
    "open",
)
STATUS_CANCELED_PATTERNS = (
    "لغو",
    "cancelled",
    "canceled",
)

def status_from_text(window: str) -> str:
    w = cleanse_text(window).lower()
    if any(p.lower() in w for p in STATUS_FULL_PATTERNS):
        return "full"
    if any(p.lower() in w for p in STATUS_OPEN_PATTERNS):
        return "open"
    if any(p.lower() in w for p in STATUS_CANCELED_PATTERNS):
        return "canceled"
    return "unknown"

DATE_RE_ASC = re.compile(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b")
DATE_RE_FA = re.compile(r"[۰-۹]{1,2}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{2,4}|[۰-۹]{4}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{1,2}")
TIME_RE = re.compile(r"(?:ساعت\s*)?(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm|AM|PM))\b")

def extract_kind(block: str) -> str:
    b = block.upper()
    if "UKVI" in b:
        return "UKVI"
    if "CDIELTS" in b or "COMPUTER-DELIVERED" in b or "COMPUTER DELIVERED" in b:
        return "CDIELTS"
    if "IELTS" in b or "آیلتس" in block:
        return "IELTS"
    return ""

# ------------------- parsers -------------------
def parse_generic_text(html: str, source: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = cleanse_text(soup.get_text(separator=" "))
    dates = []
    for m in DATE_RE_ASC.finditer(text):
        dates.append((m.group(), m.start(), m.end()))
    for m in DATE_RE_FA.finditer(text):
        dates.append((fa_to_en_digits(m.group()), m.start(), m.end()))

    entries = {}
    if dates:
        for d, s, e in dates:
            start = max(0, s - 260)
            end = min(len(text), e + 260)
            window = text[start:end]
            status = status_from_text(window)
            kind = extract_kind(window) or ("CDIELTS" if "computer" in window.lower() else "")
            tm = TIME_RE.search(window)
            time_str = tm.group(1) if tm else ""
            key = " | ".join(x for x in [source, d, time_str, kind] if x)
            entries[key] = {
                "status": status,
                "source": source,
                "date": d,
                "time": time_str or "-",
                "kind": kind or "-",
                "context": window[:300],
            }
    else:
        # fallback: single-page status
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        key = f"{source} | {h}"
        entries[key] = {
            "status": status_from_text(text),
            "source": source,
            "date": "-",
            "time": "-",
            "kind": "-",
            "context": text[:300],
        }
    return entries

def parse_irsafam(html: str) -> dict:
    return parse_generic_text(html, "Irsafam")

def parse_ieltstehran(html: str) -> dict:
    return parse_generic_text(html, "IELTS Tehran")

# ------------------- state -------------------
def load_state(path: str) -> dict:
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"entries": {}}
    return {"entries": {}}

def save_state(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def diff(prev: dict, curr: dict) -> list:
    changes = []
    prev_map = prev.get("entries", {}) if "entries" in prev else prev
    curr_map = curr.get("entries", {}) if "entries" in curr else curr
    keys = set(prev_map.keys()) | set(curr_map.keys())
    for k in sorted(keys):
        pv = prev_map.get(k)
        cv = curr_map.get(k)
        if pv is None:
            changes.append({"key": k, "type": "added", "from": None, "to": cv["status"], "entry": cv})
        elif cv is None:
            changes.append({"key": k, "type": "removed", "from": pv["status"], "to": None, "entry": pv})
        elif pv["status"] != cv["status"]:
            changes.append({"key": k, "type": "changed", "from": pv["status"], "to": cv["status"], "entry": cv})
    return changes

# ------------------- network -------------------
_http_session: aiohttp.ClientSession | None = None

async def fetch_text(url: str) -> str:
    assert _http_session is not None
    async with _http_session.get(
        url,
        headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "availability-bot/1.0"},
        timeout=aiohttp.ClientTimeout(total=25, connect=10),
    ) as r:
        return await r.text()

async def scrape_both() -> dict:
    html1, html2 = await asyncio.gather(fetch_text(URL_IRSAFAM), fetch_text(URL_TEHRAN))
    e1 = parse_irsafam(html1)
    e2 = parse_ieltstehran(html2)
    merged = {**e1, **e2}
    return merged

# ------------------- embeds -------------------
def build_embed(entries: dict) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="AAA — IELTS Timetables", timestamp=now)
    # Group by source and status, and show times explicitly
    buckets = {"Irsafam": {"open": [], "full": [], "other": []},
               "IELTS Tehran": {"open": [], "full": [], "other": []}}
    for k, v in entries.items():
        src = v.get("source", "Irsafam")
        status = v.get("status", "unknown")
        row = f"{v.get('date','-')} — {v.get('time','-')} — {v.get('kind','-')} ({status})"
        if status == "open":
            buckets.setdefault(src, {"open": [], "full": [], "other": []})["open"].append(row)
        elif status == "full":
            buckets.setdefault(src, {"open": [], "full": [], "other": []})["full"].append(row)
        else:
            buckets.setdefault(src, {"open": [], "full": [], "other": []})["other"].append(row)

    for src in ("Irsafam", "IELTS Tehran"):
        open_list = buckets[src]["open"][:20] or ["—"]
        full_list = buckets[src]["full"][:20] or ["—"]
        other_list = buckets[src]["other"][:10]
        e.add_field(name=f"{src} — Open", value="\n".join(open_list), inline=False)
        e.add_field(name=f"{src} — Full", value="\n".join(full_list), inline=False)
        if other_list:
            e.add_field(name=f"{src} — Other", value="\n".join(other_list), inline=False)

    e.url = URL_IRSAFAM
    e.set_footer(text="Ephemeral snapshot • Times shown explicitly")
    return e

def build_open_alert_embed(open_changes: list) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="Seats Available — Alert", timestamp=now, color=0x57F287)
    lines = []
    for chg in open_changes:
        v = chg["entry"]
        lines.append(f"{v.get('source','?')} — {v.get('date','-')} {v.get('time','-')} — {v.get('kind','-')}")
    e.description = "\n".join(lines[:40]) if lines else "Open slots detected."
    e.add_field(name="Sources", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    return e

# ------------------- polling / alerts -------------------
@tasks.loop(minutes=POLL_INTERVAL_MINUTES)
async def poll():
    try:
        current = await scrape_both()
    except Exception:
        return
    state = load_state(STATE_FILE)
    previous = state.get("entries", {})
    changes = diff({"entries": previous}, {"entries": current})
    if changes:
        save_state(STATE_FILE, {"entries": current})

        # Prepare alert if anything turned open
        open_changes = [c for c in changes if c["to"] == "open"]
        if open_changes:
            ch = client.get_channel(CHANNEL_ID)
            if ch is None:
                try:
                    ch = await client.fetch_channel(CHANNEL_ID)
                except Exception:
                    ch = None
            if ch:
                embed = build_open_alert_embed(open_changes)
                await ch.send(
                    content="@everyone",
                    embed=embed,
                    allowed_mentions=discord.AllowedMentions(everyone=True),
                )

# ------------------- slash command -------------------
@tree.command(name="timetable", description="Show IELTS timetable times and availability (Irsafam + IELTS Tehran)")
async def timetable(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        entries = await scrape_both()
    except Exception:
        await interaction.followup.send("Failed to fetch timetables.", ephemeral=True)
        return
    embed = build_embed(entries)
    await interaction.followup.send(embed=embed, ephemeral=True)

# ------------------- lifecycle -------------------
@client.event
async def on_ready():
    global _http_session
    if _http_session is None or _http_session.closed:
        _http_session = aiohttp.ClientSession()
    if not poll.is_running():
        poll.start()
    try:
        if GUILD_ID > 0:
            await tree.sync(guild=discord.Object(id=GUILD_ID))
        else:
            await tree.sync()
    except Exception:
        pass

# ------------------- web server + keepalive -------------------
async def _run_http():
    port = int(os.environ.get("PORT", "8000"))
    cfg = uvicorn.Config(api, host="0.0.0.0", port=port, log_level="info")
    srv = uvicorn.Server(cfg)
    await srv.serve()

async def _keep_render_awake():
    url = os.environ.get("RENDER_EXTERNAL_URL")
    if not url:
        return
    if not url.endswith("/"):
        url = url + "/"
    endpoint = url + "health"
    timeout = aiohttp.ClientTimeout(total=10)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        while True:
            try:
                await s.get(endpoint)
            except Exception:
                pass
            await asyncio.sleep(240)

# ------------------- main -------------------
async def main():
    await asyncio.gather(
        _run_http(),
        _keep_render_awake(),
        client.start(TOKEN),
    )

if __name__ == "__main__":
    asyncio.run(main())
