import os
import asyncio
import json
import re
import hashlib
from datetime import datetime, timezone

import aiohttp
import discord
from discord.ext import tasks, commands
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
URL_IRSAFAM = os.getenv("TIMETABLE_URL", "https://irsafam.org/ielts/timetable")
URL_TEHRAN = os.getenv("TIMETABLE_URL_2", "https://ieltstehran.com/computer-delivered-ielts-exam/")
POLL_INTERVAL_SECONDS = int(os.getenv("POLL_INTERVAL_SECONDS", str(int(os.getenv("POLL_INTERVAL_MINUTES", "1")) * 60)))

intents = discord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents, allowed_mentions=discord.AllowedMentions(everyone=True))
tree = bot.tree

_http: aiohttp.ClientSession | None = None
_last_snapshot: dict = {}
_poll_lock = asyncio.Lock()

STATUS_FULL_PATTERNS = ("ظرفیت: تکمیل", "ظرفیت تکمیل", "تکمیل ظرفیت", "full", "closed", "no seats", "sold out")
STATUS_OPEN_PATTERNS = ("ظرفیت: موجود", "ظرفیت موجود", "available", "open", "ثبت نام", "ثبت‌نام", "رزرو", "reserve", "book")
STATUS_CANCELED_PATTERNS = ("لغو", "cancelled", "canceled")

DATE_RE_ASC = re.compile(r"\b(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}|\d{4}[/.-]\d{1,2}[/.-]\d{1,2})\b")
DATE_RE_FA = re.compile(r"[۰-۹]{1,2}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{2,4}|[۰-۹]{4}[/\.-][۰-۹]{1,2}[/\.-][۰-۹]{1,2}")
TIME_RE = re.compile(r"(?:ساعت\s*)?(\d{1,2}:\d{2}|\d{1,2}\s*(?:am|pm|AM|PM))\b")

def _ws(t: str) -> str:
    return re.sub(r"\s+", " ", t.strip())

def _fa2en(s: str) -> str:
    return s.translate(str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789"))

def _clean(t: str) -> str:
    t = _fa2en(t)
    t = t.replace("٫", ":").replace("ː", ":").replace("،", ",")
    return _ws(t)

def _status(win: str) -> str:
    w = _clean(win).lower()
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

def _kind(block: str) -> str:
    b = block.upper()
    if "UKVI" in b:
        return "UKVI"
    if "CDIELTS" in b or "COMPUTER-DELIVERED" in b or "COMPUTER DELIVERED" in b:
        return "CDIELTS"
    if "IELTS" in b or "آیلتس" in block:
        return "IELTS"
    return "-"

async def _fetch(url: str) -> str:
    assert _http is not None
    q = "&" if "?" in url else "?"
    bust = f"{q}t={int(datetime.now().timestamp())}"
    hdrs = {"Cache-Control": "no-cache", "Pragma": "no-cache", "User-Agent": "ielts-availability-bot/1.2", "Accept-Language": "fa,en;q=0.8"}
    async with _http.get(url + bust, headers=hdrs, timeout=aiohttp.ClientTimeout(total=25, connect=10)) as r:
        return await r.text()

def _entries_from_text(text: str, source: str) -> dict:
    dates = []
    for m in DATE_RE_ASC.finditer(text):
        dates.append((m.group(), m.start(), m.end()))
    for m in DATE_RE_FA.finditer(text):
        dates.append((_fa2en(m.group()), m.start(), m.end()))
    ent = {}
    if dates:
        for d, s, e in dates:
            start = max(0, s - 180)
            end = min(len(text), e + 180)
            win = text[start:end]
            st = _status(win)
            kd = _kind(win)
            tm = TIME_RE.search(win)
            tstr = tm.group(1) if tm else "-"
            key = " | ".join(x for x in [source, d, tstr, kd] if x)
            ent[key] = {"status": st, "source": source, "date": d, "time": tstr, "kind": kd, "context": win[:300]}
    else:
        h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        key = f"{source} | {h}"
        ent[key] = {"status": _status(text), "source": source, "date": "-", "time": "-", "kind": "-", "context": text[:300]}
    return ent

def _parse_irsafam(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = _clean(soup.get_text(separator=" "))
    return _entries_from_text(text, "Irsafam")

def _parse_ieltstehran(html: str) -> dict:
    soup = BeautifulSoup(html, "lxml")
    text = _clean(soup.get_text(separator=" "))
    ent = {}
    for m in re.finditer(r"ظرفیت\s*[:：]\s*([^\s،,]+)", text):
        cap = m.group(1)
        start = max(0, m.start() - 160)
        end = min(len(text), m.end() + 160)
        win = text[start:end]
        kd = _kind(win) or "CDIELTS"
        tm = TIME_RE.search(win)
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
        if ("تکمیل" in cap_l) or ("full" in cap_l):
            st = "full"
        elif ("موجود" in cap_l) or ("available" in cap_l) or ("open" in cap_l):
            st = "open"
        else:
            st = _status(win)
            if st == "open" and ("تکمیل" in win or "ظرفیت تکمیل" in win or "تکمیل ظرفیت" in win or "full" in win or "closed" in win):
                st = "full"
        key = " | ".join(x for x in ["IELTS Tehran", date_val, tstr, kd] if x)
        ent[key] = {"status": st, "source": "IELTS Tehran", "date": date_val, "time": tstr, "kind": kd, "context": win[:300]}
    if ent:
        return ent
    return _entries_from_text(text, "IELTS Tehran")

async def scrape() -> dict:
    h1, h2 = await asyncio.gather(_fetch(URL_IRSAFAM), _fetch(URL_TEHRAN))
    e1 = _parse_irsafam(h1)
    e2 = _parse_ieltstehran(h2)
    m = {**e1, **e2}
    return m

def _load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"entries": {}}
    return {"entries": {}}

def _save_state(obj: dict) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def _diff(prev: dict, curr: dict) -> list:
    changes = []
    p = prev.get("entries", {}) if "entries" in prev else prev
    c = curr.get("entries", {}) if "entries" in curr else curr
    keys = set(p.keys()) | set(c.keys())
    for k in sorted(keys):
        pv = p.get(k)
        cv = c.get(k)
        if pv is None:
            changes.append({"key": k, "type": "added", "from": None, "to": cv["status"], "entry": cv})
        elif cv is None:
            changes.append({"key": k, "type": "removed", "from": pv["status"], "to": None, "entry": pv})
        elif pv["status"] != cv["status"]:
            changes.append({"key": k, "type": "changed", "from": pv["status"], "to": cv["status"], "entry": cv})
    return changes

def _field_txt(lines: list[str]) -> str:
    if not lines:
        return "—"
    s = "\n".join(lines)
    return s[:1024] if len(s) > 1024 else s

class Links(discord.ui.View):
    def __init__(self):
        super().__init__(timeout=None)
        self.add_item(discord.ui.Button(label="Irsafam", url=URL_IRSAFAM))
        self.add_item(discord.ui.Button(label="IELTS Tehran", url=URL_TEHRAN))

def embed_timetable(entries: dict) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="AAA — IELTS Schedules", timestamp=now, color=0x2B2D31)
    groups = {"Irsafam": {"open": [], "full": []}, "IELTS Tehran": {"open": [], "full": []}}
    for v in entries.values():
        src = v.get("source", "Irsafam")
        st = v.get("status", "unknown")
        row = f"{v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}"
        if st == "open":
            groups.setdefault(src, {"open": [], "full": []})["open"].append(f"🟢 {row}")
        elif st == "full":
            groups.setdefault(src, {"open": [], "full": []})["full"].append(f"🔴 {row}")
    for src in ("Irsafam", "IELTS Tehran"):
        e.add_field(name=f"{src} — Open", value=_field_txt(groups[src]["open"][:20]), inline=False)
        e.add_field(name=f"{src} — Full", value=_field_txt(groups[src]["full"][:20]), inline=False)
    e.add_field(name="Links", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    e.set_footer(text="Ephemeral")
    return e

def embed_alert(open_changes: list) -> discord.Embed:
    now = datetime.now(timezone.utc)
    e = discord.Embed(title="Seats Available — Alert", timestamp=now, color=0x57F287)
    lines = []
    for chg in open_changes:
        v = chg["entry"]
        lines.append(f"{v.get('source','?')} • {v.get('date','-')} • {v.get('time','-')} • {v.get('kind','-')}")
    e.description = "\n".join(lines[:40]) if lines else "Open slots detected."
    e.add_field(name="Links", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    return e

def embed_panel(entries: dict, last_run: datetime | None) -> discord.Embed:
    now = datetime.now(timezone.utc)
    total_open = sum(1 for v in entries.values() if v.get("status") == "open")
    total_full = sum(1 for v in entries.values() if v.get("status") == "full")
    e = discord.Embed(title="AAA — IELTS Bot Panel", timestamp=now, color=0x5865F2)
    e.add_field(name="Counts", value=f"🟢 Open: **{total_open}**\n🔴 Full: **{total_full}**", inline=True)
    e.add_field(name="Commands", value="`/ielts timetable` • `/ielts panel` • `/ielts refresh`", inline=True)
    if last_run:
        e.add_field(name="Last Check", value=f"{last_run.isoformat()}Z", inline=False)
    e.add_field(name="Links", value=f"[Irsafam]({URL_IRSAFAM}) • [IELTS Tehran]({URL_TEHRAN})", inline=False)
    return e

@tasks.loop(seconds=POLL_INTERVAL_SECONDS)
async def poll():
    async with _poll_lock:
        try:
            current = await scrape()
        except Exception:
            return
        prev_state = _load_state()
        prev = prev_state.get("entries", {})
        changes = _diff({"entries": prev}, {"entries": current})
        if changes:
            _save_state({"entries": current})
            open_changes = [c for c in changes if c["to"] == "open"]
            if open_changes:
                ch = bot.get_channel(CHANNEL_ID)
                if ch is None:
                    try:
                        ch = await bot.fetch_channel(CHANNEL_ID)
                    except Exception:
                        ch = None
                if ch:
                    await ch.send(content="@everyone", embed=embed_alert(open_changes), allowed_mentions=discord.AllowedMentions(everyone=True), view=Links())
        global _last_snapshot
        _last_snapshot = current

ielts = app_commands.Group(name="ielts", description="IELTS tools")

@ielts.command(name="timetable", description="Times and availability from Irsafam + IELTS Tehran")
async def cmd_timetable(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    async with _poll_lock:
        try:
            data = await scrape()
            global _last_snapshot
            _last_snapshot = data
            await interaction.followup.send(embed=embed_timetable(data), ephemeral=True, view=Links())
        except Exception:
            await interaction.followup.send("Fetch failed.", ephemeral=True)

@ielts.command(name="panel", description="Dashboard and quick links")
async def cmd_panel(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    data = _last_snapshot or _load_state().get("entries", {})
    await interaction.followup.send(embed=embed_panel(data, datetime.now(timezone.utc)), ephemeral=True, view=Links())

@ielts.command(name="refresh", description="Force an immediate scrape")
async def cmd_refresh(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True, thinking=True)
    async with _poll_lock:
        try:
            data = await scrape()
            _save_state({"entries": data})
            global _last_snapshot
            _last_snapshot = data
            await interaction.followup.send("Refreshed.", ephemeral=True)
        except Exception:
            await interaction.followup.send("Refresh failed.", ephemeral=True)

tree.add_command(ielts)

@bot.event
async def on_ready():
    global _http
    if _http is None or _http.closed:
        _http = aiohttp.ClientSession()
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
    await asyncio.gather(_run_http(), _keep_render_awake(), bot.start(TOKEN))

if __name__ == "__main__":
    asyncio.run(main())
