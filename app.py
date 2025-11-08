import os
import asyncio
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, List

import aiohttp
import discord
from discord.ext import tasks, commands
from discord import app_commands
from bs4 import BeautifulSoup
from fastapi import FastAPI
import uvicorn

api = FastAPI()

@api.get("/")
async def root():
    return {"ok": True}

@api.get("/health")
async def health():
    return {"ok": True}

TOKEN = os.environ["DISCORD_TOKEN"]
CHANNEL_ID = int(os.environ["DISCORD_CHANNEL_ID"])
GUILD_ID = int(os.getenv("GUILD_ID", "0"))
STATE_FILE = os.getenv("STATE_FILE", "state.json")
URL_IRSAFAM = os.getenv("TIMETABLE_URL", "https://irsafam.org/ielts/timetable")
URL_TEHRAN = os.getenv("TIMETABLE_URL_2", "https://ieltstehran.com/computer-delivered-ielts-exam/")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", "45"))

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=discord.AllowedMentions(everyone=True))
tree = bot.tree

_http: Optional[aiohttp.ClientSession] = None
_last_snapshot: Dict[str, dict] = {}
_poll_lock = asyncio.Lock()

STATUS_FULL_TOKENS = ("ظرفیت: تکمیل", "ظرفیت تکمیل", "تکمیل ظرفیت", "full", "closed", "sold out", "no seats")
STATUS_OPEN_TOKENS = ("ظرفیت: موجود", "ظرفیت موجود", "available", "open")
STATUS_CANCEL_TOKENS = ("لغو", "cancelled", "canceled")

DATE_RE_ASC = re.compile(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b")
DATE_RE_FA  = re.compile(r"[۰-۹]{1,2}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{2,4}|[۰-۹]{4}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{1,2}")
TIME_RE     = re.compile(r"(?:ساعت\s*)?(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm|AM|PM))\b", re.I)
CAP_RE      = re.compile(r"ظرفیت\s*[:：]\s*([^\s،,]+)")

def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def _fa2en(s: str) -> str:
    return s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))

def _clean(t: str) -> str:
    t = _fa2en(t)
    t = t.replace("٫", ":").replace("ː", ":").replace("،", ",")
    return _ws(t)

def _status_generic(win: str) -> str:
    w = _clean(win).lower()
    if any(tok.lower() in w for tok in STATUS_FULL_TOKENS):
        return "full"
    if any(tok.lower() in w for tok in STATUS_CANCEL_TOKENS):
        return "canceled"
    if any(tok.lower() in w for tok in STATUS_OPEN_TOKENS):
        return "open"
    return "unknown"

def _kind(block: str) -> str:
    b = block.upper()
    if "UKVI" in b:
        return "UKVI"
    if "CDIELTS" in b or "COMPUTER-DELIVERED" in b or "COMPUTER DELIVERED" in b:
        return "CDIELTS"
    if "IELTS" in b or "آیلتس" in block:
        return "IELTS"
    return "-"

async def ensure_http() -> aiohttp.ClientSession:
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=20, connect=8))
    return _http

async def _fetch(url: str) -> str:
    s = await ensure_http()
    bust = ("&" if "?" in url else "?") + f"t={int(datetime.now().timestamp())}"
    hdrs = {"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "ielts-availability-bot/1.5", "Accept-Language": "fa,en;q=0.8"}
    for _ in range(3):
        try:
            async with s.get(url + bust, headers=hdrs) as r:
                if r.status == 200:
                    return await r.text()
        except Exception:
            await asyncio.sleep(0.3)
    return ""

def _entries_from_text(text: str, source: str) -> Dict[str, dict]:
    dates: List[tuple] = []
    for m in DATE_RE_ASC.finditer(text):
        dates.append((m.group(), m.start(), m.end()))
    for m in DATE_RE_FA.finditer(text):
        dates.append((_fa2en(m.group()), m.start(), m.end()))
    ent: Dict[str, dict] = {}
    if dates:
        for d, s, e in dates:
            start = max(0, s - 140)
            end   = min(len(text), e + 140)
            win = text[start:end]
            st  = _status_generic(win)
            kd  = _kind(win)
            tm  = TIME_RE.search(win)
            tstr = tm.group(1) if tm else "-"
            key = " | ".join(x for x in [source, d, tstr, kd] if x)
            ent[key] = {"status": st, "source": source, "date": d, "time": tstr, "kind": kd, "context": win[:300]}
    else:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        key = f"{source} | {h}"
        ent[key] = {"status": _status_generic(text), "source": source, "date": "-", "time": "-", "kind": "-", "context": text[:300]}
    return ent

def parse_irsafam(html: str) -> Dict[str, dict]:
    soup = BeautifulSoup(html, "lxml")
    text = _clean(soup.get_text(separator=" "))
    return _entries_from_text(text, "Irsafam")

def parse_ieltstehran(html: str) -> Dict[str, dict]:
    soup = BeautifulSoup(html, "lxml")
    text = _clean(soup.get_text(separator=" "))
    ent: Dict[str, dict] = {}
    for m in CAP_RE.finditer(text):
        cap = m.group(1)
        start = max(0, m.start() - 140)
        end   = min(len(text), m.end() + 140)
        win = text[start:end]
        kd  = _kind(win) or "CDIELTS"
        tm  = TIME_RE.search(win)
        tstr = tm.group(1) if tm else "-"
        date_val = "-"
        best = None
        for dm in DATE_RE_ASC.finditer(win):
            dist = abs((start + dm.start()) - m.start())
            if best is None or dist < best[0]:
                best = (dist, dm.group())
        if best is None:
            for dm in DATE_RE_FA.finditer(win):
                dist = abs((start + dm.start()) - m.start())
                if best is None or dist < best[0]:
                    best = (dist, _fa2en(dm.group()))
        if best is not None:
            date_val = best[1]
        cap_l = cap.lower()
        if "تکمیل" in cap_l or "full" in cap_l or "closed" in cap_l:
            st = "full"
        elif "موجود" in cap_l or "open" in cap_l or "available" in cap_l or "خالی" in cap_l or "باز" in cap_l:
            st = "open"
        else:
            st = "unknown"
        key = " | ".join(x for x in ["IELTS Tehran", date_val, tstr, kd] if x)
        ent[key] = {"status": st, "source": "IELTS Tehran", "date": date_val, "time": tstr, "kind": kd, "context": win[:300]}
    if ent:
        return ent
    return _entries_from_text(text, "IELTS Tehran")

async def scrape_both() -> Dict[str, dict]:
    h1, h2 = await asyncio.gather(_fetch(URL_IRSAFAM), _fetch(URL_TEHRAN))
    e1 = parse_irsafam(h1 or "")
    e2 = parse_ieltstehran(h2 or "")
    return {**e1, **e2}

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"entries": {}}
    return {"entries": {}}

def save_state(obj: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def diff(prev: dict, curr: dict) -> List[dict]:
    p = prev.get("entries", {}) if "entries" in prev else prev
    c = curr.get("entries", {}) if "entries" in curr else curr
    keys = set(p.keys()) | set(c.keys())
    out: List[dict] = []
    for k in keys:
        pv = p.get(k)
        cv = c.get(k)
        if pv is None:
            out.append({"key": k, "type": "added", "from": None, "to": cv["status"], "entry": cv})
        elif cv is None:
            out.append({"key": k, "type": "removed", "from": pv["status"], "to": None, "entry": pv})
        elif pv["status"] != cv["status"]:
            out.append({"key": k, "type": "changed", "from": pv["status"], "to": cv["status"], "entry": cv})
    return out

def _field_text(lines: List[str]) -> str:
    if not lines:
        return "—"
    s = "\n".join(lines)
    return s[:1024] if len(s) > 1024 else s

class SourceLinks(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="Irsafam", url=URL_IRSAFAM))
        self.add_item(discord.ui.Button(label="IELTS Tehran", url=URL_TEHRAN))

def embed_timetable(entries: Dict[str, dict]) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="IELTS — Times & Availability", timestamp=now, color=0x2B2D31)
    groups = {"Irsafam": {"open": [], "full": []}, "IELTS Tehran": {"open": [], "full": []}}
    for v in entries.values():
        src = v.get("source", "Irsafam")
        st  = v.get("status", "unknown")
        row = f"{v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}"
        if st == "open":
            groups.setdefault(src, {"open": [], "full": []})["open"].append(f"🟢 {row}")
        elif st == "full":
            groups.setdefault(src, {"open": [], "full": []})["full"].append(f"🔴 {row}")
    for src in ("Irsafam", "IELTS Tehran"):
        e.add_field(name=f"{src} — Open", value=_field_text(groups[src]["open"][:20]), inline=False)
        e.add_field(name=f"{src} — Full", value=_field_text(groups[src]["full"][:20]), inline=False)
    e.add_field(name="Sources", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    e.set_footer(text="Ephemeral snapshot")
    return e

def embed_alert(open_changes: List[dict]) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="Seats Available — Alert", timestamp=now, color=0x57F287)
    lines = []
    for chg in open_changes:
        v = chg["entry"]
        lines.append(f"{v.get('source','?')} • {v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}")
    e.description = "\n".join(lines[:40]) if lines else "Open slots detected."
    e.add_field(name="Links", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    return e

def embed_panel(entries: Dict[str, dict]) -> discord.Embed:
    now = datetime.now(timezone.utc)
    total_open = sum(1 for v in entries.values() if v.get("status") == "open")
    total_full = sum(1 for v in entries.values() if v.get("status") == "full")
    e = discord.Embed(title="IELTS Bot — Panel", timestamp=now, color=0x5865F2)
    e.add_field(name="Open", value=f"🟢 **{total_open}**", inline=True)
    e.add_field(name="Full", value=f"🔴 **{total_full}**", inline=True)
    e.add_field(name="Slash Commands", value="`/ielts timetable` • `/ielts panel` • `/ielts refresh`", inline=False)
    e.add_field(name="Auto Announcements", value="Tags @everyone when seats open.", inline=False)
    return e

def _ephemeral_for(interaction: discord.Interaction) -> bool:
    return interaction.guild_id is not None

async def _ensure_initial_response(interaction: discord.Interaction, ephemeral: bool) -> None:
    if interaction.response.is_done():
        return
    try:
        await interaction.response.defer(ephemeral=ephemeral, thinking=True)
    except Exception:
        try:
            await interaction.response.send_message("\u200b", ephemeral=ephemeral)
        except Exception:
            pass

async def _safe_edit_original(interaction: discord.Interaction, *, content: Optional[str] = None, embed: Optional[discord.Embed] = None, view: Optional[discord.ui.View] = None, ephemeral: bool) -> None:
    try:
        await interaction.edit_original_response(content=content, embed=embed, view=view)
    except Exception:
        try:
            if interaction.response.is_done():
                await interaction.followup.send(content=content, embed=embed, view=view, ephemeral=ephemeral)
            else:
                await interaction.response.send_message(content=content, embed=embed, view=view, ephemeral=ephemeral)
        except Exception:
            pass

@tasks.loop(seconds=POLL_INTERVAL_SECONDS)
async def poll():
    async with _poll_lock:
        try:
            current = await scrape_both()
        except Exception:
            return
        state = load_state()
        prev = state.get("entries", {})
        changes = diff({"entries": prev}, {"entries": current})
        if changes:
            save_state({"entries": current})
            open_changes = [c for c in changes if c["to"] == "open"]
            if open_changes:
                ch = bot.get_channel(CHANNEL_ID)
                if ch is None:
                    try:
                        ch = await bot.fetch_channel(CHANNEL_ID)
                    except Exception:
                        ch = None
                if ch:
                    await ch.send(content="@everyone", embed=embed_alert(open_changes), allowed_mentions=discord.AllowedMentions(everyone=True), view=SourceLinks())
        global _last_snapshot
        _last_snapshot = current

ielts = app_commands.Group(name="ielts", description="IELTS availability tools")

async def _refresh_and_edit(interaction: discord.Interaction):
    global _last_snapshot
    ephemeral = _ephemeral_for(interaction)
    try:
        async with _poll_lock:
            fresh = await asyncio.wait_for(scrape_both(), timeout=8)
            _last_snapshot = fresh
        await _safe_edit_original(interaction, embed=embed_timetable(fresh), view=SourceLinks(), ephemeral=ephemeral)
    except Exception:
        pass

@ielts.command(name="timetable", description="Show times and availability")
async def cmd_timetable(interaction: discord.Interaction):
    ephemeral = _ephemeral_for(interaction)
    await _ensure_initial_response(interaction, ephemeral)
    data = _last_snapshot or load_state().get("entries", {})
    await _safe_edit_original(interaction, embed=embed_timetable(data), view=SourceLinks(), ephemeral=ephemeral)
    asyncio.create_task(_refresh_and_edit(interaction))

@ielts.command(name="panel", description="Dashboard and quick links")
async def cmd_panel(interaction: discord.Interaction):
    ephemeral = _ephemeral_for(interaction)
    await _ensure_initial_response(interaction, ephemeral)
    data = _last_snapshot or load_state().get("entries", {})
    await _safe_edit_original(interaction, embed=embed_panel(data), view=SourceLinks(), ephemeral=ephemeral)

@ielts.command(name="refresh", description="Force an immediate scrape")
async def cmd_refresh(interaction: discord.Interaction):
    global _last_snapshot
    ephemeral = _ephemeral_for(interaction)
    await _ensure_initial_response(interaction, ephemeral)
    try:
        async with _poll_lock:
            data = await asyncio.wait_for(scrape_both(), timeout=10)
            save_state({"entries": data})
            _last_snapshot = data
        await _safe_edit_original(interaction, content="Refreshed.", embed=None, view=None, ephemeral=ephemeral)
    except Exception:
        await _safe_edit_original(interaction, content="Refresh failed.", embed=None, view=None, ephemeral=ephemeral)

tree.add_command(ielts)

@bot.event
async def on_ready():
    await ensure_http()
    if not poll.is_running():
        poll.start()
    try:
        if GUILD_ID > 0:
            await tree.sync(guild=discord.Object(id=GUILD_ID))
        else:
            for g in bot.guilds:
                await tree.sync(guild=g)
            await tree.sync()
    except Exception as e:
        print(f"Sync error: {e}")

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
    s = await ensure_http()
    while True:
        try:
            await s.get(endpoint)
        except Exception:
            pass
        await asyncio.sleep(240)

async def main():
    await asyncio.gather(_run_http(), _keep_render_awake(), bot.start(TOKEN))

if __name__ == "__main__":
    asyncio.run(main())
