import os
import asyncio
import json
import re
import hashlib
import aiohttp
import discord
from discord.ext import tasks
from discord import app_commands
from bs4 import BeautifulSoup
from fastapi import FastAPI
import uvicorn
from datetime import datetime, timezone

api = FastAPI()

@api.get("/health")
async def health():
    return {"ok": True}

URL = os.getenv("TIMETABLE_URL", "https://irsafam.org/ielts/timetable")
TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["DISCORD_CHANNEL_ID"])
STATE_FILE = os.getenv("STATE_FILE", "state.json")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))
GUILD_ID = int(os.getenv("GUILD_ID", "0"))

intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

def normalize_text(t):
    return re.sub(r"\s+", " ", t.strip())

def fa_to_en_digits(s):
    trans = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
    return s.translate(trans)

def status_from_text(t):
    tn = normalize_text(t).lower()
    if ("تکمیل ظرفیت" in tn) or ("ظرفیت تکمیل" in tn) or ("full" in tn) or ("closed" in tn):
        return "full"
    if ("ثبت نام" in tn) or ("ثبت‌نام" in tn) or ("available" in tn) or ("open" in tn):
        return "open"
    if ("لغو" in tn) or ("cancelled" in tn) or ("canceled" in tn):
        return "canceled"
    return "unknown"

def parse_page(html):
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    text = normalize_text(text)
    dates = []
    patt_ascii = re.compile(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b")
    patt_fa = re.compile(r"[۰-۹]{1,2}[/\.-][۰-۹]{1,2}[/\. -][۰-۹]{2,4}|[۰-۹]{4}[/\. -][۰-۹]{1,2}[/\. -][۰-۹]{1,2}")
    for m in patt_ascii.finditer(text):
        dates.append((m.group(), m.start(), m.end()))
    for m in patt_fa.finditer(text):
        dates.append((fa_to_en_digits(m.group()), m.start(), m.end()))
    entries = {}
    if dates:
        for d, s, e in dates:
            start = max(0, s - 260)
            end = min(len(text), e + 260)
            window = text[start:end]
            status = status_from_text(window)
            kind = ""
            if "UKVI" in window.upper():
                kind = "UKVI"
            elif "CDIELTS" in window.upper():
                kind = "CDIELTS"
            elif ("IELTS" in window.upper()) or ("آیلتس" in window):
                kind = "IELTS"
            time_match = re.search(r"\b\d{1,2}:\d{2}\b", window)
            time_str = time_match.group(0) if time_match else ""
            key = " ".join(x for x in [kind, d, time_str] if x).strip()
            if key not in entries or entries[key]["status"] == "unknown":
                entries[key] = {"status": status, "context": window[:300]}
    else:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        st = status_from_text(text)
        entries[h] = {"status": st, "context": text[:300]}
    return entries

def load_state(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except:
                return {"entries": {}}
    return {"entries": {}}

def save_state(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def diff(prev, curr):
    changes = []
    prev_keys = set(prev.keys())
    curr_keys = set(curr.keys())
    for k in sorted(prev_keys | curr_keys):
        pv = prev.get(k)
        cv = curr.get(k)
        if pv is None:
            changes.append({"key": k, "type": "added", "from": None, "to": cv["status"], "context": cv["context"]})
        elif cv is None:
            changes.append({"key": k, "type": "removed", "from": pv["status"], "to": None, "context": pv.get("context", "")})
        elif pv["status"] != cv["status"]:
            changes.append({"key": k, "type": "changed", "from": pv["status"], "to": cv["status"], "context": cv["context"]})
    return changes

def build_embed(entries: dict) -> discord.Embed:
    e = discord.Embed(title="AAA — IELTS Timetable", url=URL, timestamp=datetime.now(timezone.utc))
    open_items = [(k, v) for k, v in entries.items() if v.get("status") == "open"]
    full_items = [(k, v) for k, v in entries.items() if v.get("status") == "full"]
    other_items = [(k, v) for k, v in entries.items() if v.get("status") not in ("open", "full")]
    desc = []
    if open_items:
        desc.append("Open")
        for k, _ in sorted(open_items)[:15]:
            desc.append(f"• {k}")
    if full_items:
        if desc:
            desc.append("")
        desc.append("Full")
        for k, _ in sorted(full_items)[:15]:
            desc.append(f"• {k}")
    if other_items:
        if desc:
            desc.append("")
        desc.append("Other")
        for k, v in sorted(other_items)[:10]:
            desc.append(f"• {k}: {v.get('status','unknown')}")
    e.description = "\n".join(desc) if desc else "No timetable entries found."
    e.set_footer(text="Ephemeral snapshot")
    return e

async def fetch_html_fresh():
    timeout = aiohttp.ClientTimeout(total=25, connect=10)
    async with aiohttp.ClientSession(timeout=timeout, headers={"Cache-Control": "no-cache", "Pragma": "no-cache"}) as session:
        async with session.get(URL) as r:
            return await r.text()

@tasks.loop(minutes=POLL_INTERVAL_MINUTES)
async def poll():
    html = await fetch_html_fresh()
    current = parse_page(html)
    state = load_state(STATE_FILE)
    prev = state.get("entries", {})
    changes = diff(prev, current)
    if changes or not prev:
        ch = client.get_channel(CHANNEL_ID)
        if ch is None:
            try:
                ch = await client.fetch_channel(CHANNEL_ID)
            except Exception:
                ch = None
        if ch and changes:
            lines = []
            for c in changes:
                if c["type"] == "added":
                    lines.append(f"[ADDED] {c['key']} -> {c['to']}")
                elif c["type"] == "removed":
                    lines.append(f"[REMOVED] {c['key']} (was {c['from']})")
                else:
                    lines.append(f"[CHANGED] {c['key']}: {c['from']} -> {c['to']}")
                lines.append(f"{c['context']}")
                lines.append("")
            msg = "\n".join(lines).strip() + f"\nSource: {URL}"
            await ch.send(msg)
        save_state(STATE_FILE, {"entries": current})

@tree.command(name="timetable", description="AAA — show current IELTS timetable and availability")
async def timetable(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    html = await fetch_html_fresh()
    entries = parse_page(html)
    await interaction.followup.send(embed=build_embed(entries), ephemeral=True)

@client.event
async def on_ready():
    if not poll.is_running():
        poll.start()
    try:
        if GUILD_ID > 0:
            await tree.sync(guild=discord.Object(id=GUILD_ID))
        else:
            await tree.sync()
    except Exception:
        pass

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

async def main():
    await asyncio.gather(
        _run_http(),
        _keep_render_awake(),
        client.start(TOKEN),
    )

if __name__ == "__main__":
    asyncio.run(main())
