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

api = FastAPI()

@api.get("/health")
async def health():
    return {"ok": True}

TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["DISCORD_CHANNEL_ID"])
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")
POLL_INTERVAL_MINUTES = int(os.getenv("POLL_INTERVAL_MINUTES", "5"))
URL_IRSAFAM = os.getenv("TIMETABLE_URL", "https://irsafam.org/ielts/timetable")
URL_TEHRAN = os.getenv("TIMETABLE_URL_2", "https://ieltstehran.com/computer-delivered-ielts-exam/")

intents = discord.Intents.default()
client = discord.Client(intents=intents, allowed_mentions=discord.AllowedMentions(everyone=True))
tree = app_commands.CommandTree(client)

def normalize_ws(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def fa_to_en_digits(s: str) -> str:
    return s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))

def cleanse_text(t: str) -> str:
    t = fa_to_en_digits(t)
    t = t.replace("٫", ":").replace("ː", ":").replace("،", ",")
    return normalize_ws(t)

STATUS_FULL_PATTERNS = ("ظرفیت: تکمیل", "ظرفیت تکمیل", "تکمیل ظرفیت", "full", "closed", "no seats", "sold out")
STATUS_OPEN_PATTERNS = ("ظرفیت: موجود", "ظرفیت موجود", "available", "open", "ثبت نام", "ثبت‌نام", "رزرو", "reserve", "book")
STATUS_CANCELED_PATTERNS = ("لغو", "cancelled", "canceled")

def status_from_text(window: str) -> str:
    w = cleanse_text(window).lower()
    full_hit = any(p.lower() in w for p in STATUS_FULL_PATTERNS)
    open_hit = any(p.lower() in w for p in STATUS_OPEN_PATTERNS)
    cancel_hit = any(p.lower() in w for p in STATUS_CANCELED_PATTERNS)
    if full_hit and open_hit:
        return "full"
    if full_hit:
        return "full"
    if cancel_hit:
        return "canceled"
    if open_hit and not full_hit:
        return "open"
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
    return "-"

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
            start = max(0, s - 180)
            end = min(len(text), e + 180)
            window = text[start:end]
            status = status_from_text(window)
            kind = extract_kind(window)
            tm = TIME_RE.search(window)
            time_str = tm.group(1) if tm else "-"
            key = " | ".join(x for x in [source, d, time_str, kind] if x)
            entries[key] = {"status": status, "source": source, "date": d, "time": time_str, "kind": kind, "context": window[:300]}
    else:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        key = f"{source} | {h}"
        entries[key] = {"status": status_from_text(text), "source": source, "date": "-", "time": "-", "kind": "-", "context": text[:300]}
    return entries

def parse_irsafam(html: str) -> dict:
    return parse_generic_text(html, "Irsafam")

def parse_ieltstehran(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = cleanse_text(soup.get_text(separator=" "))
    entries = {}
    for m in re.finditer(r"ظرفیت\s*[:]\s*([^\s،,]+)", text):
        cap = m.group(1)
        start = max(0, m.start() - 200)
        end = min(len(text), m.end() + 200)
        window = text[start:end]
        date_val = "-"
        nearest = None
        for dm in DATE_RE_ASC.finditer(window):
            dist = abs((start + dm.start()) - m.start())
            if nearest is None or dist < nearest[0]:
                nearest = (dist, dm.group())
        if nearest is None:
            for dm in DATE_RE_FA.finditer(window):
                dist = abs((start + dm.start()) - m.start())
                if nearest is None or dist < nearest[0]:
                    nearest = (dist, fa_to_en_digits(dm.group()))
        if nearest is not None:
            date_val = nearest[1]
        tm = TIME_RE.search(window)
        time_str = tm.group(1) if tm else "-"
        kind = extract_kind(window)
        cap_l = cap.lower()
        if ("تکمیل" in cap_l) or ("full" in cap_l):
            st = "full"
        elif ("موجود" in cap_l) or ("available" in cap_l) or ("open" in cap_l):
            st = "open"
        else:
            st = status_from_text(window)
        key = " | ".join(x for x in ["IELTS Tehran", date_val, time_str, kind] if x)
        entries[key] = {"status": st, "source": "IELTS Tehran", "date": date_val, "time": time_str, "kind": kind, "context": window[:300]}
    if entries:
        return entries
    return parse_generic_text(html, "IELTS Tehran")

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

_http_session: aiohttp.ClientSession | None = None

async def fetch_text(url: str) -> str:
    assert _http_session is not None
    async with _http_session.get(url, headers={"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "availability-bot/1.1"}, timeout=aiohttp.ClientTimeout(total=25, connect=10)) as r:
        return await r.text()

async def scrape_both() -> dict:
    html1, html2 = await asyncio.gather(fetch_text(URL_IRSAFAM), fetch_text(URL_TEHRAN))
    e1 = parse_irsafam(html1)
    e2 = parse_ieltstehran(html2)
    merged = {**e1, **e2}
    return merged

def _field_text(lines: list[str]) -> str:
    if not lines:
        return "—"
    s = "\n".join(lines)
    return s[:1024] if len(s) > 1024 else s

class SourceLinks(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="Irsafam", url=URL_IRSAFAM))
        self.add_item(discord.ui.Button(label="IELTS Tehran", url=URL_TEHRAN))

def build_embed(entries: dict) -> discord.Embed:
    now = datetime.now(timezone.utc)
    color = 0x2B2D31
    e = discord.Embed(title="IELTS Schedules — Times & Availability", timestamp=now, color=color)
    groups = {"Irsafam": {"open": [], "full": []}, "IELTS Tehran": {"open": [], "full": []}}
    for v in entries.values():
        src = v.get("source", "Irsafam")
        status = v.get("status", "unknown")
        row = f"{v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}"
        if status == "open":
            groups.setdefault(src, {"open": [], "full": []})["open"].append(f"🟢 {row}")
        elif status == "full":
            groups.setdefault(src, {"open": [], "full": []})["full"].append(f"🔴 {row}")
    for src in ("Irsafam", "IELTS Tehran"):
        e.add_field(name=f"{src} — Open", value=_field_text(groups[src]["open"][:20]), inline=False)
        e.add_field(name=f"{src} — Full", value=_field_text(groups[src]["full"][:20]), inline=False)
    e.add_field(name="Sources", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    e.set_footer(text="Ephemeral snapshot")
    return e

def build_open_alert_embed(open_changes: list) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="Seats Available — Alert", timestamp=now, color=0x57F287)
    lines = []
    for chg in open_changes:
        v = chg["entry"]
        lines.append(f"{v.get('source','?')} • {v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}")
    e.description = "\n".join(lines[:40]) if lines else "Open slots detected."
    e.add_field(name="Links", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    return e

def build_panel_embed(entries: dict) -> discord.Embed:
    now = datetime.now(timezone.utc)
    total_open = sum(1 for v in entries.values() if v.get("status") == "open")
    total_full = sum(1 for v in entries.values() if v.get("status") == "full")
    e = discord.Embed(title="IELTS Bot — Panel", timestamp=now, color=0x5865F2)
    e.add_field(name="Totals", value=f"🟢 Open: **{total_open}**\n🔴 Full: **{total_full}**", inline=True)
    e.add_field(name="Commands", value="`/ielts timetable` — show times & availability\n`/ielts panel` — overview & links", inline=True)
    e.add_field(name="Announcements", value="Auto-alerts tag @everyone when seats open.", inline=False)
    return e

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
                await ch.send(content="@everyone", embed=embed, allowed_mentions=discord.AllowedMentions(everyone=True), view=SourceLinks())

ielts = app_commands.Group(name="ielts", description="IELTS tools")

@ielts.command(name="timetable", description="Show IELTS timetable times and availability")
async def cmd_timetable(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        entries = await scrape_both()
    except Exception:
        await interaction.followup.send("Failed to fetch timetables.", ephemeral=True)
        return
    await interaction.followup.send(embed=build_embed(entries), ephemeral=True, view=SourceLinks())

@ielts.command(name="panel", description="Bot panel, counts, links")
async def cmd_panel(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    try:
        entries = await scrape_both()
    except Exception:
        await interaction.followup.send("Failed to fetch data.", ephemeral=True)
        return
    await interaction.followup.send(embed=build_panel_embed(entries), ephemeral=True, view=SourceLinks())

tree.add_command(ielts)

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
    await asyncio.gather(_run_http(), _keep_render_awake(), client.start(TOKEN))

if __name__ == "__main__":
    asyncio.run(main())
