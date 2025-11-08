import os
import asyncio
import json
import re
import hashlib
from datetime import datetime, timezone
from typing import Optional, Dict, List

import logging
import aiohttp
import discord
from discord.ext import tasks, commands
from discord import app_commands
from bs4 import BeautifulSoup
from fastapi import FastAPI
import uvicorn

logging.basicConfig(level=logging.INFO)
discord.utils.setup_logging(level=logging.INFO)

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
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=discord.AllowedMentions(everyone=True))
tree = bot.tree

_http: Optional[aiohttp.ClientSession] = None
_last_snapshot: Dict[str, dict] = {}
_panel_message_id: Optional[int] = None
_poll_lock = asyncio.Lock()

STATUS_FULL_TOKENS = (
    "ظرفیت: تکمیل", "ظرفیت تکمیل", "تکمیل ظرفیت", "تکمیل", "پر", "ناموجود",
    "full", "closed", "sold out", "no seats", "not available"
)
STATUS_OPEN_TOKENS = (
    "ظرفیت: موجود", "ظرفیت موجود", "خالی", "باز", "در دسترس", "قابل رزرو",
    "ثبت نام", "رزرو", "available", "open", "book now", "register"
)

MONTHS = r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
DATE_RE_ASC = re.compile(r"\b(?:\d{1,2}[/\.-]\d{1,2}[/\.-]\d{2,4}|\d{4}[/\.-]\d{1,2}[/\.-]\d{1,2})\b", re.I)
DATE_RE_MON1 = re.compile(rf"\b(?:\d{{1,2}}\s+(?:{MONTHS})\s+\d{{4}})\b", re.I)
DATE_RE_MON2 = re.compile(rf"\b(?:(?:{MONTHS})\s+\d{{1,2}},?\s+\d{{4}})\b", re.I)
DATE_RE_FA  = re.compile(r"[۰-۹]{1,2}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{2,4}|[۰-۹]{4}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{1,2}")
TIME_RE     = re.compile(r"(?:ساعت\s*)?(\d{1,2}[:٫]\d{2}|\d{1,2}\s*(?:am|pm|AM|PM))\b", re.I)
FULL_RE     = re.compile(r"(تکمیل(?:\s*ظرفیت)?|ظرفیت\s*تکمیل|ناموجود|full|closed|sold out|no seats)", re.I)
OPEN_RE     = re.compile(r"(ظرفیت\s*(?:موجود|باز|خالی)|در\s*دسترس|قابل\s*رزرو|ثبت\s*نام|رزرو|available|open|book\s*now|register)", re.I)

def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def _fa2en(s: str) -> str:
    return s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")).replace("٫", ":")

def _clean(t: str) -> str:
    t = _fa2en(t)
    t = t.replace("\u200c", " ").replace("\u200f", " ").replace("\u202a", " ").replace("\u202b", " ").replace("،", ",")
    return _ws(t)

def _kind(block: str) -> str:
    b = block.upper()
    if "UKVI" in b or "CDUKVI" in b:
        return "UKVI"
    if "CDIELTS" in b or "COMPUTER-DELIVERED" in b or "COMPUTER DELIVERED" in b or "CD " in b or "CD-" in b:
        return "CDIELTS"
    if "IELTS" in b or "آیلتس" in block:
        return "IELTS"
    return "-"

async def ensure_http() -> aiohttp.ClientSession:
    global _http
    if _http is None or _http.closed:
        connector = aiohttp.TCPConnector(limit=6, ttl_dns_cache=600, ssl=False, keepalive_timeout=45)
        timeout = aiohttp.ClientTimeout(total=45, connect=15, sock_read=30, sock_connect=15)
        _http = aiohttp.ClientSession(connector=connector, timeout=timeout, trust_env=True)
    return _http

async def _fetch(url: str) -> str:
    s = await ensure_http()
    bust = ("&" if "?" in url else "?") + f"t={int(datetime.now().timestamp())}"
    origin = url.split("/", 3)
    referer = f"{origin[0]}//{origin[2]}/" if len(origin) >= 3 else url
    hdrs = {
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
        "Accept-Language": "fa,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": referer,
        "DNT": "1",
    }
    delay = 0.7
    for _ in range(6):
        try:
            async with s.get(url + bust, headers=hdrs, allow_redirects=True) as r:
                if r.status == 200:
                    text = await r.text(errors="ignore")
                    if text and len(text) > 500:
                        return text
        except Exception:
            await asyncio.sleep(delay)
            delay = min(4.0, delay * 1.5)
    return ""

def _pick_date_from_node(node_text: str) -> str:
    t = _clean(node_text)
    for rx in (DATE_RE_ASC, DATE_RE_MON1, DATE_RE_MON2, DATE_RE_FA):
        m = rx.search(t)
        if m:
            g = m.group(0)
            return _fa2en(g)
    return "-"

def _pick_time_from_node(node_text: str) -> str:
    t = _clean(node_text)
    m = TIME_RE.search(t)
    return _fa2en(m.group(1)) if m else "-"

def _scan_blocks_for_status(soup: BeautifulSoup, source: str) -> Dict[str, dict]:
    ent: Dict[str, dict] = {}
    def capture(block_text: str, status: str):
        kd = _kind(block_text)
        dstr = _pick_date_from_node(block_text)
        tstr = _pick_time_from_node(block_text)
        key = " | ".join(x for x in [source, dstr, tstr, kd] if x)
        ent[key] = {"status": status, "source": source, "date": dstr, "time": tstr, "kind": kd, "context": _clean(block_text)[:300]}
    for tag in soup.find_all(text=FULL_RE):
        blk = tag
        for _ in range(6):
            if getattr(blk, "parent", None) is None:
                break
            blk = blk.parent
            txt = blk.get_text(separator=" ", strip=True)
            if len(txt) > 40:
                capture(txt, "full")
                break
    for tag in soup.find_all(text=OPEN_RE):
        blk = tag
        for _ in range(6):
            if getattr(blk, "parent", None) is None:
                break
            blk = blk.parent
            txt = blk.get_text(separator=" ", strip=True)
            if len(txt) > 40:
                capture(txt, "open")
                break
    return ent

def parse_irsafam(html: str) -> Dict[str, dict]:
    soup = BeautifulSoup(html, "lxml")
    ent = _scan_blocks_for_status(soup, "Irsafam")
    if ent:
        return ent
    text = soup.get_text(separator=" ")
    return _entries_from_text(text, "Irsafam")

def parse_ieltstehran(html: str) -> Dict[str, dict]:
    soup = BeautifulSoup(html, "lxml")
    ent = _scan_blocks_for_status(soup, "IELTS Tehran")
    if ent:
        return ent
    text = soup.get_text(separator=" ")
    return _entries_from_text(text, "IELTS Tehran")

def _entries_from_text(text: str, source: str) -> Dict[str, dict]:
    t = _clean(text)
    ent: Dict[str, dict] = {}
    for m in FULL_RE.finditer(t):
        start = max(0, m.start() - 220)
        end = min(len(t), m.end() + 220)
        win = t[start:end]
        kd = _kind(win)
        dstr = _pick_date_from_node(win)
        tstr = _pick_time_from_node(win)
        key = " | ".join(x for x in [source, dstr, tstr, kd] if x)
        ent[key] = {"status": "full", "source": source, "date": dstr, "time": tstr, "kind": kd, "context": win[:300]}
    for m in OPEN_RE.finditer(t):
        start = max(0, m.start() - 220)
        end = min(len(t), m.end() + 220)
        win = t[start:end]
        kd = _kind(win)
        dstr = _pick_date_from_node(win)
        tstr = _pick_time_from_node(win)
        key = " | ".join(x for x in [source, dstr, tstr, kd] if x)
        ent[key] = {"status": "open", "source": source, "date": dstr, "time": tstr, "kind": kd, "context": win[:300]}
    if ent:
        return ent
    snippet = " ".join(t.split()[:80])
    if snippet:
        h = hashlib.sha256(snippet.encode("utf-8")).hexdigest()[:16]
        key = f"{source} | {h}"
        ent[key] = {"status": "full", "source": source, "date": "-", "time": "-", "kind": "-", "context": snippet[:300]}
    return ent

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
            return {"entries": {}, "panel_message_id": None}
    return {"entries": {}, "panel_message_id": None}

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
        if pv is None and cv is not None:
            out.append({"key": k, "type": "added", "from": None, "to": cv["status"], "entry": cv})
        elif cv is None and pv is not None:
            out.append({"key": k, "type": "removed", "from": pv["status"], "to": None, "entry": pv})
        elif pv and cv and pv.get("status") != cv.get("status"):
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
    groups = {
        "Irsafam": {"open": [], "full": []},
        "IELTS Tehran": {"open": [], "full": []},
    }
    for v in entries.values():
        src = v.get("source", "Irsafam")
        st  = "open" if v.get("status") == "open" else "full"
        row = f"{v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}"
        if st == "open":
            groups.setdefault(src, {"open": [], "full": []})["open"].append(f"🟢 {row}")
        else:
            groups.setdefault(src, {"open": [], "full": []})["full"].append(f"🔴 {row}")
    for src in ("Irsafam", "IELTS Tehran"):
        e.add_field(name=f"{src} — Open", value=_field_text(groups[src]["open"][:20]), inline=False)
        e.add_field(name=f"{src} — Full", value=_field_text(groups[src]["full"][:20]), inline=False)
    e.add_field(name="Sources", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    e.set_footer(text="Auto-refresh snapshot")
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
    total_full = sum(1 for v in entries.values() if v.get("status") != "open")
    e = discord.Embed(title="IELTS Bot — Panel", timestamp=now, color=0x5865F2)
    e.add_field(name="Open", value=f"🟢 **{total_open}**", inline=True)
    e.add_field(name="Full", value=f"🔴 **{total_full}**", inline=True)
    e.add_field(name="Slash Commands", value="`/timetable` • `/panel`", inline=False)
    e.add_field(name="Auto Announcements", value="Tags @everyone when seats open.", inline=False)
    return e

def _load_ids_from_state() -> None:
    global _panel_message_id
    st = load_state()
    _panel_message_id = st.get("panel_message_id")

async def _save_ids_to_state() -> None:
    st = load_state()
    st["panel_message_id"] = _panel_message_id
    st["entries"] = _last_snapshot
    save_state(st)

async def _ensure_panel_message() -> Optional[discord.Message]:
    ch = bot.get_channel(CHANNEL_ID)
    if ch is None:
        try:
            ch = await bot.fetch_channel(CHANNEL_ID)
        except Exception:
            return None
    global _panel_message_id
    if _panel_message_id:
        try:
            msg = await ch.fetch_message(_panel_message_id)
            return msg
        except Exception:
            _panel_message_id = None
    try:
        msg = await ch.send(embed=embed_timetable(_last_snapshot or {}), view=SourceLinks())
        _panel_message_id = msg.id
        await _save_ids_to_state()
        return msg
    except Exception:
        return None

async def _update_panel_message() -> None:
    msg = await _ensure_panel_message()
    if msg is None:
        return
    try:
        await msg.edit(embed=embed_timetable(_last_snapshot or {}), view=SourceLinks())
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
            save_state({"entries": current, "panel_message_id": state.get("panel_message_id")})
            open_changes = [c for c in changes if c["to"] == "open"]
            if open_changes:
                ch = bot.get_channel(CHANNEL_ID)
                if ch is None:
                    try:
                        ch = await bot.fetch_channel(CHANNEL_ID)
                    except Exception:
                        ch = None
                if ch:
                    try:
                        await ch.send(
                            content="@everyone",
                            embed=embed_alert(open_changes),
                            allowed_mentions=discord.AllowedMentions(everyone=True),
                            view=SourceLinks(),
                        )
                    except Exception:
                        pass
        global _last_snapshot
        _last_snapshot = current
        await _update_panel_message()

async def handle_timetable(interaction: discord.Interaction):
    data = _last_snapshot or load_state().get("entries", {})
    try:
        await interaction.response.send_message(embed=embed_timetable(data), view=SourceLinks(), ephemeral=True)
    except Exception:
        pass

async def handle_panel(interaction: discord.Interaction):
    data = _last_snapshot or load_state().get("entries", {})
    try:
        await interaction.response.send_message(embed=embed_panel(data), view=SourceLinks(), ephemeral=True)
    except Exception:
        pass

def register_commands():
    guild_obj = discord.Object(id=GUILD_ID) if GUILD_ID > 0 else None

    async def _timetable(interaction: discord.Interaction):
        await handle_timetable(interaction)

    async def _panel(interaction: discord.Interaction):
        await handle_panel(interaction)

    if guild_obj:
        tree.command(name="timetable", description="Show times and availability", guild=guild_obj)(_timetable)
        tree.command(name="panel", description="Dashboard and quick links", guild=guild_obj)(_panel)
    else:
        tree.command(name="timetable", description="Show times and availability")(_timetable)
        tree.command(name="panel", description="Dashboard and quick links")(_panel)

register_commands()

@tree.error
async def on_app_command_error(interaction: discord.Interaction, error: app_commands.AppCommandError):
    pass

@bot.event
async def on_ready():
    _load_ids_from_state()
    await ensure_http()
    if not poll.is_running():
        poll.start()
    try:
        for g in bot.guilds:
            try:
                await tree.sync(guild=g)
            except Exception:
                pass
        await tree.sync()
        if GUILD_ID > 0:
            try:
                await tree.sync(guild=discord.Object(id=GUILD_ID))
            except Exception:
                pass
    except Exception:
        pass
    try:
        async with _poll_lock:
            snap = await scrape_both()
            if snap:
                global _last_snapshot
                _last_snapshot = snap
        await _save_ids_to_state()
        await _update_panel_message()
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
    s = await ensure_http()
    while True:
        try:
            await s.get(endpoint)
        except Exception:
            pass
        await asyncio.sleep(240)

async def _run_bot():
    await bot.start(TOKEN)

async def main():
    await asyncio.gather(_run_http(), _keep_render_awake(), _run_bot())

if __name__ == "__main__":
    asyncio.run(main())
