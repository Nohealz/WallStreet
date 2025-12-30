import discord
import io
import csv
from PIL import Image, ImageStat, ImageEnhance
import pytesseract
import re
import os
from decouple import config
import numpy as np
import cv2
import sqlite3
from datetime import datetime, timedelta, time
import asyncio  # Add this to use asyncio functionalities
import subprocess, ast
import sys
from statistics import mean

from zoneinfo import ZoneInfo
EASTERN_TZ = ZoneInfo("America/New_York")
RUN_AT_ET = time(15, 55, 0)  # 3:55pm Eastern
    
# Discord bot token and channel IDs
AUTHORIZED_USER = int(config('AUTHORIZED_USER'))
DISCORD_TOKEN = config('DISCORD_TOKEN')
SOURCE_CHANNEL_ID = int(config('SOURCE_CHANNEL_ID'))  # Channel for posting screenshots
# EOD Screener config
ALERT_CHANNEL_ID = int(config('ALERT_CHANNEL_ID'))
SERVER_ID = int(config('SERVER_ID'))

# absolute path to your finviz screener
FINVIZ_SCRIPT_PATH = config('FINVIZ_SCRIPT_PATH')

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
client = discord.Client(intents=intents)

class BotState:
    def __init__(self):
        self.relay_enabled = True      # existing behavior
        self.finviz_today = []         # today’s Finviz tickers (from _run_finviz_and_notify)
        self.finviz_task = None        # scheduler task handle to avoid duplicate schedules

DEFAULT_SIM_CONF = 96  # default confidence for text-input tests / VAL|NN

def _parse_token_with_conf(token: str):
    """
    Accept either 'VAL' or 'VAL|NN' where NN is an int confidence.
    Returns (text, conf_int).
    """
    if "|" in token:
        val, conf = token.rsplit("|", 1)
        try:
            c = int(conf)
        except:
            c = DEFAULT_SIM_CONF
        return val, c
    return token, DEFAULT_SIM_CONF

@client.event
async def on_ready():
    print(f"{client.user} has connected to Discord!")
    # Avoid spawning multiple scheduler tasks if on_ready fires after reconnects.
    if bot_state.finviz_task is None or bot_state.finviz_task.done():
        bot_state.finviz_task = client.loop.create_task(finviz_daily_scheduler())
    else:
        print("finviz scheduler already running; skip re-create.")

@client.event
async def on_message(message):
    # Ignore the bot's own messages
    if message.author == client.user:
        return

    # Define a helper function to log and notify unauthorized access
    async def log_unauthorized_access(action, user):
        # Log the unauthorized attempt in CMD
        print(f"Unauthorized {action} attempt by user {user.name} (ID: {user.id}) in server: {message.guild.name} (ID: {message.guild.id})")

        # Notify the source channel
        source_channel = await client.fetch_channel(SOURCE_CHANNEL_ID)
        if source_channel:
            await source_channel.send(
                f"@here Unauthorized {action} attempt by user {user.name} (ID: {user.id}) in server: {message.guild.name} (ID: {message.guild.id})"
            )
    
    # Handle !bind command
    if message.content.lower().startswith("!bind"):
        if message.author.id != AUTHORIZED_USER:
            await log_unauthorized_access("bind", message.author)
            await message.channel.send("You are not authorized to use this command.")
            return
        parts = message.content.split(maxsplit=1)
        description = parts[1] if len(parts) > 1 else "No description"

        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        try:
            # Insert or update the binding with the description
            cursor.execute('''
                INSERT OR REPLACE INTO bindings (server_id, channel_id, description)
                VALUES (?, ?, ?)
            ''', (str(message.guild.id), str(message.channel.id), description))
            conn.commit()
            await message.channel.send(f"Channel bound with description: '{description}'")
        except Exception as e:
            await message.channel.send(f"Failed to bind the channel: {e}")
        finally:
            conn.close()

    # Handle !unbind command
    elif message.content.lower().startswith("!unbind"):
        if message.author.id != AUTHORIZED_USER:
            await log_unauthorized_access("unbind", message.author)
            await message.channel.send("You are not authorized to use this command.")
            return
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        try:
            cursor.execute('DELETE FROM bindings WHERE server_id = ? AND channel_id = ?',
                           (str(message.guild.id), str(message.channel.id)))
            conn.commit()
            await message.channel.send("Channel unbound successfully.")
        except Exception as e:
            await message.channel.send(f"Failed to unbind the channel: {e}")
        finally:
            conn.close()

    # Handle !list_bindings command
    elif message.content.lower().startswith("!list_bindings"):
        if message.author.id != AUTHORIZED_USER:
            await log_unauthorized_access("list_bindings", message.author)
            await message.channel.send("You are not authorized to use this command.")
            return
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        try:
            cursor.execute('SELECT server_id, channel_id, description FROM bindings')
            rows = cursor.fetchall()
            if rows:
                response = "**Current Bindings:**\n```\n"
                for server_id, channel_id, description in rows:
                    response += f"Server: {server_id}, Channel: {channel_id}, Description: {description}\n"
                response += "```"
            else:
                response = "No channels are currently bound."
            await message.channel.send(response)
        except Exception as e:
            await message.channel.send(f"Failed to list bindings: {e}")
        finally:
            conn.close()

    # Handle !text command
    elif message.content.strip().lower().startswith("!text"):
        if message.author.id != AUTHORIZED_USER:  # Ensure only the authorized user can run this command
            await message.channel.send("You are not authorized to use this command.")
            return

        # Extract the message content after "!text"
        broadcast_message = message.content[len("!text"):].strip()
        if not broadcast_message:
            await message.channel.send("Please provide a message to broadcast.")
            return

        # Relay the message to all bound channels
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        cursor.execute('SELECT server_id, channel_id FROM bindings')
        bound_channels = cursor.fetchall()
        conn.close()

        
        if bot_state.relay_enabled:
            for server_id, channel_id in bound_channels:
                channel = client.get_channel(int(channel_id))
                if channel:
                    try:
                        await channel.send(broadcast_message)
                        print(f"Broadcasted message to channel {channel_id} in server {server_id}.")
                    except Exception as e:
                        print(f"Failed to send message to channel {channel_id} in server {server_id}: {e}")
    
            await message.channel.send("Message broadcasted to all bound channels.")
        else:
            await message.channel.send("bot off. Message **NOT** broadcasted to all bound channels.")

    # Handle !delete command
    elif message.content.strip().lower() == "!delete":
        if message.author.id != AUTHORIZED_USER:  # Ensure only the authorized user can run this command
            await message.channel.send("You are not authorized to use this command.")
            return

        # Fetch all bound channels
        conn = sqlite3.connect('trades.db')
        cursor = conn.cursor()
        cursor.execute('SELECT server_id, channel_id FROM bindings')
        bound_channels = cursor.fetchall()
        conn.close()

        for server_id, channel_id in bound_channels:
            channel = client.get_channel(int(channel_id))
            if channel:
                try:
                    # Fetch the last message sent by the bot in the channel
                    async for msg in channel.history(limit=10):  # Check the last 10 messages
                        if msg.author == client.user:  # Find a message from the bot
                            await msg.delete()
                            print(f"Deleted the last message in channel {channel_id} in server {server_id}.")
                            break
                except Exception as e:
                    print(f"Failed to delete message in channel {channel_id} in server {server_id}: {e}")

        await message.channel.send("Last bot message deleted from all bound channels.")

    # Check if the message is in the source channel
    if message.channel.id == SOURCE_CHANNEL_ID:
        print(f"Received message: {message.content}, ID: {message.id}")  # Debug log
        # Get the source channel
        source_channel = client.get_channel(SOURCE_CHANNEL_ID)
        
        # Handle !help command
        if message.content.strip().lower() == "!help":
            help_message = """
**Available Commands:**
- `!on` — Enable relaying messages to bound channels.
- `!off` — Disable relaying messages to bound channels.
- `!text` — Broadcast a message to all bound channels.
- `!positions` – Display all current positions.
- `!summary [YYYY-MM-DD] [--local]` – Daily summary for the given date (default: today). `--local` skips the alert channel and posts EOD summary only to the source channel.
- `!fees` – Upload broker activity CSV/TSV to import short borrow fees (admin only).
- `!eod` – Post the EOD strategy summary (open holdings + closed-trade metrics) to the alert channel.
- `!bind <description>` — Bind the current channel for relaying messages (admin only).
- `!unbind` — Unbind the current channel (admin only).
- `!list_bindings` — Show all bound channels (admin only).
- `!delete` — Delete the bot’s last message in all bound channels.
- `!help` — Show this command list.
- **Upload an image** — Attach a trade screenshot; the bot will OCR, validate, prompt for fixes if needed, then log trades.

**Manual trade entry (no screenshot):**
- `!trade` — Starts a prompt. Reply with one or more lines:
  - `SYMBOL PRICE QTY` *(time defaults to `12:00:00`)*  
  - `TIME SYMBOL PRICE QTY`
  - You can simulate OCR confidence with `value|NN` (e.g., `FI|87`).
- `!trade ABCD 23.70 50` — One-line quick mode on the same line (also accepts `TIME ABCD 23.70 50`).
- Flags: `--sell` (default) or `--buy` to mark action.

**Notes:**
- Symbols must be UPPERCASE. If OCR ever returns lowercase, it’s auto-flagged for correction.
- Any field with confidence ≤ 88 triggers the correction prompt.
- Prices print to 2 decimals; quantities must be integers.
            """
            await source_channel.send(help_message)
        
        # Handle !on command
        if message.content.strip().lower() == "!on":
            bot_state.relay_enabled = True
            await source_channel.send("Relay to bound channels is now enabled.")

        # Handle !off command
        elif message.content.strip().lower() == "!off":
            bot_state.relay_enabled = False
            await source_channel.send("Relay to bound channels is now disabled.")

        # Trigger the get_all_positions command
        elif message.content.strip().lower() == "!positions":
            positions_report = get_all_positions()
            # Relay to all bound channels
            if bot_state.relay_enabled:
                await relay_to_bound_channels(positions_report)
            await source_channel.send(positions_report)

        # Trigger the daily summary command
        elif message.content.startswith("!summary"):
            try:
                # Check for an optional date argument
                parts = message.content.split()
                send_alert = True
                if any(p in ("--local", "--test") for p in parts):
                    send_alert = False
                parts = [p for p in parts if not p.startswith("--")]
                if len(parts) > 1:
                    # Validate provided date
                    date_str = parts[1]
                    datetime.strptime(date_str, "%Y-%m-%d")  # Ensure valid date format
                else:
                    # Default to today's date
                    date_str = datetime.now().strftime("%Y-%m-%d")

                # Fetch and send the daily summary
                summary = get_daily_summary(date_str)
                if bot_state.relay_enabled:
                    await relay_to_bound_channels(summary)
                await source_channel.send(summary)
                # ✅ ALSO run EOD summary when !summary is run
                await post_eod_summary(client, also_send_channel_id=SOURCE_CHANNEL_ID, send_alert=send_alert)
                
            except ValueError:
                await source_channel.send("@here Invalid date format. Please use YYYY-MM-DD, e.g., `!summary 2025-01-19`.")
            except Exception as e:
                await source_channel.send(f"@here An error occurred: {str(e)}")

        elif message.content.strip().lower().startswith("!fees"):
            if message.author.id != AUTHORIZED_USER:
                await source_channel.send("You are not authorized to use this command.")
                return

            if not message.attachments:
                await source_channel.send("Attach your broker activity export (CSV/TSV) with `!fees`.")
                return

            handled = False
            for attachment in message.attachments:
                filename = attachment.filename.lower()
                if not filename.endswith((".csv", ".tsv", ".txt")):
                    continue
                handled = True
                file_path = f"./{attachment.filename}"
                await attachment.save(file_path)
                try:
                    report = import_borrow_fees_csv(file_path)
                    await source_channel.send(report)
                except Exception as e:
                    await source_channel.send(f"@here Fee import failed: {e}")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)

            if not handled:
                await source_channel.send("No CSV/TSV attachment found to import.")
        
        elif message.content.lower().startswith("!trade"):
            try:
                src = await client.fetch_channel(SOURCE_CHANNEL_ID)

                lines = message.content.splitlines()
                header = lines[0].strip()
                header_tokens = header.split()

                # Default action: short (SELL); flip with --buy
                action = "SELL"
                if "--buy" in header_tokens:
                    action = "BUY"

                # Non-flag tokens after !trade on same line → quick mode first row
                nonflags = [t for t in header_tokens[1:] if not t.startswith("--")]

                row_lines: list[str] = []

                if nonflags:
                    row_lines.append(" ".join(nonflags))
                    row_lines += [ln.strip() for ln in lines[1:] if ln.strip()]
                else:
                    # Prompted mode
                    prompt = (
                        "**Manual Trade Entry**\n"
                        "Reply with one or more lines (paste in one message):\n"
                        "```\n"
                        "SYMBOL PRICE QTY\n"
                        "SYMBOL|CONF PRICE|CONF QTY|CONF\n"
                        "TIME SYMBOL PRICE QTY\n"
                        "```\n"
                        "- Time optional → defaults to `12:00:00`\n"
                        "- Use `VAL|NN` to simulate confidence (e.g., `FI|87`)\n"
                        "- Example:\n"
                        "```\n"
                        "APLS 20.69 1\n"
                        "FI|87 70.54 14\n"
                        "09:42:10 FMC 15.55 67\n"
                        "```\n"
                        "Type `cancel` to abort."
                    )
                    await src.send(prompt)

                    def _check(m):
                        return m.channel.id == src.id and m.author == message.author

                    reply = await client.wait_for("message", timeout=300, check=_check)
                    text = reply.content.strip()
                    if text.lower() == "cancel":
                        await src.send("Manual trade entry cancelled.")
                        return

                    row_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

                # Build OCR-like rows: [(text, conf)] x 4
                bad = []
                trade_data = []
                for ln in row_lines:
                    parts = ln.split()

                    # Allow no-time form: SYMBOL PRICE QTY → prepend default time
                    if len(parts) == 3:
                        parts = ["12:00:00"] + parts

                    if len(parts) != 4:
                        bad.append(f"`{ln}` — expected SYMBOL PRICE QTY or TIME SYMBOL PRICE QTY")
                        continue

                    t_tok, s_tok, p_tok, q_tok = parts
                    t, c_t = _parse_token_with_conf(t_tok)
                    s, c_s = _parse_token_with_conf(s_tok)
                    p, c_p = _parse_token_with_conf(p_tok)
                    q, c_q = _parse_token_with_conf(q_tok)

                    trade_data.append([(t, c_t), (s, c_s), (p, c_p), (q, c_q)])

                if bad:
                    await src.send("I couldn't parse these lines:\n" + "\n".join("• " + x for x in bad))
                    return

                # Run your correction UI (same as OCR)
                trade_data = await gather_trade_row_corrections(trade_data)

                # Convert to extracted_trades expected by generate_trade_report
                extracted_trades = []
                for (t_time, _), (t_sym, _), (t_price, _), (t_qty, _) in trade_data:
                    symbol = (t_sym or "").upper()
                    try:
                        price = float(t_price)
                        qty_abs = abs(int(str(t_qty).lstrip("+")))  # pass UNSIGNED qty to the reporter
                    except Exception:
                        continue

                    extracted_trades.append({
                        "time": t_time,
                        "symbol": symbol,
                        "price": price,
                        "qty": qty_abs,     # <-- UNSIGNED here; reporter applies sign based on action
                        "action": action,   # "SELL" or "BUY"
                    })

                if not extracted_trades:
                    await src.send("No valid trades to process.")
                    return

                # DO NOT write to DB here. Keep parity with attachment flow.
                # Let generate_trade_report handle DB updates and final rendering.
                try:
                    report = generate_trade_report(extracted_trades)
                except Exception as e:
                    print(f"generate_trade_report failed: {e}")
                    # Minimal fallback if reporter fails
                    report = "\n".join(
                        ["**Trade Summary**"] +
                        [f"{tr['time']} {tr['symbol']} {tr['price']:.2f} {tr['qty']} [{tr['action']}]"
                         for tr in extracted_trades]
                    )

                # Send to source
                await src.send(report)

                # Relay if enabled (same as attachment flow)
                if bot_state.relay_enabled:
                    try:
                        await relay_to_bound_channels(report)
                    except Exception as e:
                        print(f"relay_to_bound_channels failed: {e}")

            except asyncio.TimeoutError:
                await src.send("Timed out waiting for manual trade rows. Start again with `!trade`.")
            except Exception as e:
                print(f"!trade failed: {e}")
                await src.send(f"@here `!trade` failed: {e}")



        # Handle image attachments for trades
        elif message.attachments:            
            for attachment in message.attachments:
                if attachment.filename.lower().endswith(('png', 'jpg', 'jpeg')):
                    image_path = f"./{attachment.filename}"
                    await attachment.save(image_path)
        
                    try:
                        # Process the image and get debug data
                        result = await process_image(image_path)
                        if not result:
                            await source_channel.send("@here I couldn't detect a valid table header in that screenshot. Try a clearer crop.")
                            continue  # move to next attachment or exit loop

                        # Guard #2: missing keys
                        if not isinstance(result, dict) or "trades" not in result:
                            await source_channel.send("@here I parsed the image but couldn't find a trades table.")
                            continue

                        trades = result["trades"]

                        # Guard #3: empty trades list
                        if not trades:
                            await source_channel.send("@here I found the header but no trade rows were readable.")
                            continue

                        ocr_text = result["ocr_text"]
                        confidences = result["confidences"]
        
                        # Get the source channel (assume it's already defined as SOURCE_CHANNEL_ID)
                        source_channel = client.get_channel(SOURCE_CHANNEL_ID)
                        
                        if bot_state.relay_enabled:
                            # In relay ON mode, generate the trade report, relay to bound channels, etc.
                            report = generate_trade_report(trades)
                            await relay_to_bound_channels(report)
                            await source_channel.send(report)
                        else:
                            # Relay is OFF: send the raw debug info to the source channel
                            debug_message = "**DEBUG INFO (Relay OFF)**\n"
                            debug_message += "**Trades:**\n" + f"```{trades}```\n"
                            debug_message += "**OCR Text:**\n" + f"```{ocr_text}```\n"
                            debug_message += "**Confidences:**\n" + f"```{confidences}```"
                            await source_channel.send(debug_message)
                    finally:
                        if os.path.exists(image_path):
                            os.remove(image_path)

def initialize_db():
    """
    Initialize the SQLite database with all required tables.
    """
    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()

    # === Core trades table ===
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trades (
            symbol TEXT PRIMARY KEY,
            total_qty INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            realized_pl REAL DEFAULT 0
        )
    ''')

    # === Trade history (per execution record) ===
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trade_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            qty INTEGER NOT NULL,
            price REAL NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    # === Discord channel bindings for trade reports ===
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS bindings (
            server_id TEXT NOT NULL,
            channel_id TEXT NOT NULL,
            description TEXT DEFAULT "No description",
            PRIMARY KEY (server_id, channel_id)
        )
    ''')

    # === EOD Shorts strategy table ===
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS eod_shorts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            total_qty INTEGER NOT NULL,
            avg_cost REAL NOT NULL,
            short_fees REAL DEFAULT 0,
            realized_pl REAL DEFAULT 0,
            date_first_open TEXT NOT NULL, -- MM/DD/YY (original open date)
            date_start TEXT NOT NULL,   -- MM/DD/YY
            date_closed TEXT            -- MM/DD/YY or NULL
        )
    ''')

    # Add short_fees for existing DBs that were created before this column existed
    cursor.execute("PRAGMA table_info(eod_shorts)")
    cols = [row[1] for row in cursor.fetchall()]
    if "short_fees" not in cols:
        cursor.execute("ALTER TABLE eod_shorts ADD COLUMN short_fees REAL DEFAULT 0")
    if "date_first_open" not in cols:
        cursor.execute("ALTER TABLE eod_shorts ADD COLUMN date_first_open TEXT")
        cursor.execute("UPDATE eod_shorts SET date_first_open = date_start WHERE date_first_open IS NULL")

    # === Borrow fee history (deduped across uploads) ===
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS short_fee_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            fee_date TEXT NOT NULL,        -- YYYY-MM-DD
            amount REAL NOT NULL,
            description TEXT,
            imported_at TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS uniq_short_fee
        ON short_fee_history(symbol, fee_date, amount, description)
    ''')
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_short_fee_symbol_date
        ON short_fee_history(symbol, fee_date)
    ''')

    # === Indexes for fast lookups ===
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_eod_shorts_open ON eod_shorts(symbol, date_closed)")

    conn.commit()
    conn.close()

    
def preprocess_with_opencv(image_path):
    """
    Preprocess the image using OpenCV to enhance visibility for OCR and determine text color.
    """
    # Read the image
    image = cv2.imread(image_path)

    # Split the image into color channels
    b, g, r = cv2.split(image)


    # Preprocess the image for OCR
    # Process the red channel for red text
    _, red_thresh = cv2.threshold(r, 120, 255, cv2.THRESH_BINARY)

    # Process the green channel for green text
    _, green_thresh = cv2.threshold(g, 120, 255, cv2.THRESH_BINARY)

    # Combine thresholds: keep red and green text
    combined_thresh = cv2.bitwise_or(red_thresh, green_thresh)

    # Invert the image to make text white and background black
    inverted = cv2.bitwise_not(combined_thresh)

    # Use erosion to refine text edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    eroded = cv2.erode(inverted, kernel, iterations=1)

    # Apply light dilation to slightly thicken the text
    dilated = cv2.dilate(eroded, kernel, iterations=1)

    # Save the preprocessed image for debugging
    cv2.imwrite("debug_opencv_preprocessed.png", dilated)

    return dilated

async def process_image(image_path):
    # Read and preprocess the image
    original_image = cv2.imread(image_path)
    preprocessed_image = preprocess_with_opencv(image_path)
    
    # Use Tesseract to extract bounding box data
    bounding_data = pytesseract.image_to_data(
        preprocessed_image, output_type=pytesseract.Output.DICT
    )
    print("OCR Bounding Box Data:")
    print(bounding_data)
    
    # --- Build a list of OCR entries (text, confidence) for non-empty texts ---
    ocr_text = bounding_data["text"]
    confidences = bounding_data["conf"]
    ocr_entries = [
        (text, conf)
        for text, conf in zip(ocr_text, confidences)
        if text.strip() != ""
    ]
    
    # --- Identify header row by searching for header keywords in order ---
    header_keywords = ["Time", "Symbol", "Price", "Qty"]
    header_indices = []
    start_index = 0
    for keyword in header_keywords:
        found = False
        for i in range(start_index, len(ocr_entries)):
            if ocr_entries[i][0] == keyword:
                header_indices.append(i)
                start_index = i + 1
                found = True
                break
        if not found:
            print("Header not detected. Expected keywords missing:", header_keywords)
            return None
    
    header_row = [ocr_entries[i][0] for i in header_indices]
    print("\nHeader Row detected:", header_row)
    
    # --- All entries after the header will be considered trade entries ---
    trade_entries = ocr_entries[header_indices[-1] + 1:]
    print("\nTrade Entries (raw):", trade_entries)
    
    # --- Group trade entries into rows of 4 ---
    trade_data = []
    for i in range(0, len(trade_entries), 4):
        row = trade_entries[i:i+4]
        trade_data.append(row)
    
    # --- Gather errors from trade rows and prompt for corrections via Discord ---
    trade_data = await gather_trade_row_corrections(trade_data)
    
    # --- Parse trade rows into structured trade data ---
    trades = []
    for row in trade_data:
        if len(row) == 4:
            try:
                time_val = row[0][0]
                symbol_val = row[1][0]
                price_val = float(row[2][0])
                qty_val = int(row[3][0])
                trade = {
                    "time": time_val,
                    "symbol": symbol_val,
                    "price": price_val,
                    "qty": qty_val,
                    "action": determine_row_color(
                        original_image,
                        " ".join([field[0] for field in row]),
                        bounding_data
                    )
                }
                trades.append(trade)
            except ValueError as e:
                print(f"Error parsing trade row: {row}. Skipping. Error: {e}")
        else:
            print(f"Incomplete trade row skipped: {row}")
    
    if not trades:
        print("No valid trade data extracted.")
    else:
        print(f"Extracted Trades: {trades}")
    
    # Return a dictionary with trades and the raw OCR info for debugging:
    return {
        "trades": trades,
        "ocr_text": bounding_data["text"],
        "confidences": bounding_data["conf"]
    }
    
TIME_RE   = re.compile(r"^\d{1,2}:\d{2}:\d{2}$")      # e.g., 9:05:07, 16:08:15
SYMB_RE   = re.compile(r"^[A-Z][A-Z0-9.\-]*$")        # UPPERCASE only
PRICE_RE  = re.compile(r"^\d+(?:\.\d{1,2})?$")        # up to 2 decimals
INT_RE    = re.compile(r"^[+-]?\d+$")                 # integer (allow sign)
CONF_REVIEW_MAX = 88                                   # conf 0–88 forces review

def _fmt_price(x: str) -> str:
    return f"{float(x):.2f}"


async def gather_trade_row_corrections(trade_data):
    """
    Ask the user to correct error rows in Discord.

    REQUIRED INPUT FORMAT (one row per line):
        row  time_or_?  symbol_or_?  price_or_?  qty_or_?
    Examples:
        4 16:08:15 APLS 20.69 1
        4 ? APLS ? 1
        4 ? ? ? 1
        4 ??? 1      (shorthand for row ? ? ? qty)

    Rules:
      - '?' keeps the OCR’d value.
      - lowercase OCR symbols are auto-flagged (conf=1).
      - price prints to 2 decimals.
      - invalid input -> ask again.
    """

    # ---- Normalize confidences & penalize lowercase symbols ----
    normalized = []
    for row in trade_data:
        r = list(row)
        while len(r) < 4:
            r.append(("", -1))
        fixed = []
        for text, conf in r:
            try:
                c = int(float(conf))
            except:
                c = -1
            fixed.append((text, c))

        sym, conf = fixed[1]
        if any(ch.islower() for ch in (sym or "")):
            fixed[1] = (sym, 1)  # lowercase penalty

        normalized.append(fixed)
    trade_data = normalized

    # ---- Error test ----
    def row_has_error(row):
        if len(row) != 4:
            return True

        (t_time, c_time), (t_sym, c_sym), (t_price, c_price), (t_qty, c_qty) = row

        if not TIME_RE.match(t_time or ""): return True
        if not SYMB_RE.match((t_sym or "").upper()): return True
        if not PRICE_RE.match(t_price or ""): return True
        if not INT_RE.match(t_qty or ""): return True

        if any(0 <= c <= CONF_REVIEW_MAX for c in (c_time, c_sym, c_price, c_qty)):
            return True

        return False

    error_rows = [(i, r) for i, r in enumerate(trade_data) if row_has_error(r)]
    if not error_rows:
        return trade_data

    listing = "**The following trade rows have errors:**\n"
    for idx, row in error_rows:
        listing += f"Row {idx+1}: " + ", ".join([f"{t} (conf:{c})" for t, c in row]) + "\n"

    instructions = (
        "\nCorrect format (exactly 5 tokens per line):\n"
        "`row time|? symbol|? price|? qty|?`\n\n"
        "Examples:\n"
        "`4 16:08:15 APLS 20.69 1`\n"
        "`4 ? APLS ? 1`\n"
        "`4 ? ? ? 1`\n"
        "`4 ??? 1`  (shorthand)\n\n"
        "Type `skip` to continue without fixing."
    )

    source_channel = await client.fetch_channel(SOURCE_CHANNEL_ID)

    while True:
        await source_channel.send(listing + instructions)

        def check(m):
            return m.channel.id == source_channel.id and m.author != client.user

        try:
            response = await client.wait_for("message", timeout=300, check=check)
        except asyncio.TimeoutError:
            await source_channel.send("No corrections received. Using OCR values.")
            return trade_data

        content = response.content.strip()
        if content.lower() == "skip":
            await source_channel.send("Skipping corrections. Using OCR values.")
            return trade_data

        lines = [l.strip() for l in content.split("\n") if l.strip()]
        bad = []
        updates = []

        for line in lines:
            parts = line.split()

            # shorthand: `4 ??? 1` → `4 ? ? ? 1`
            # Extra shorthand: allow ?? to mean "? ?"
            # Example: "1 ? FI ??" → "1 ? FI ? ?"
            expanded = []
            for token in parts:
                if token == "??":
                    expanded.extend(["?", "?"])
                elif token == "???":
                    expanded.extend(["?", "?", "?"])
                else:
                    expanded.append(token)
            parts = expanded

            if len(parts) != 5:
                bad.append((line, "Must be 5 tokens"))
                continue

            row_tok, time_tok, sym_tok, price_tok, qty_tok = parts

            if not row_tok.isdigit():
                bad.append((line, "Row index must be an integer"))
                continue
            idx = int(row_tok) - 1
            if not (0 <= idx < len(trade_data)):
                bad.append((line, "Row index out of range"))
                continue

            if time_tok != "?" and not TIME_RE.match(time_tok):
                bad.append((line, "Invalid time"))
                continue

            if sym_tok != "?" and not SYMB_RE.match(sym_tok.upper()):
                bad.append((line, "Symbol must be UPPERCASE or '?'"))
                continue

            if price_tok != "?" and not PRICE_RE.match(price_tok):
                bad.append((line, "Invalid price"))
                continue

            if qty_tok != "?" and not INT_RE.match(qty_tok):
                bad.append((line, "Invalid qty"))
                continue

            updates.append((idx, {
                "time": None if time_tok == "?" else time_tok,
                "symbol": None if sym_tok == "?" else sym_tok.upper(),
                "price": None if price_tok == "?" else _fmt_price(price_tok),
                "qty": None if qty_tok == "?" else qty_tok
            }))

        if bad:
            msg = "Fix these and try again:\n"
            for l, why in bad:
                msg += f"• `{l}` — {why}\n"
            await source_channel.send(msg)
            continue

        # Apply updates
        for idx, patch in updates:
            (o_time, c_time), (o_sym, c_sym), (o_price, c_price), (o_qty, c_qty) = trade_data[idx]

            if patch["price"] is None:
                try: kept_price = _fmt_price(o_price)
                except: kept_price = o_price
            else:
                kept_price = patch["price"]

            trade_data[idx] = [
                (patch["time"]   if patch["time"]   is not None else o_time,   100 if patch["time"]   is not None else c_time),
                (patch["symbol"] if patch["symbol"] is not None else o_sym,    100 if patch["symbol"] is not None else c_sym),
                (kept_price,                                                 100 if patch["price"]  is not None else c_price),
                (patch["qty"]    if patch["qty"]    is not None else o_qty,    100 if patch["qty"]    is not None else c_qty)
            ]

        # Show final
        final = "**Final Trade Rows:**\n"
        for i, row in enumerate(trade_data):
            final += f"Row {i+1}: " + ", ".join([f"{t} (conf:{c})" for t, c in row]) + "\n"
        await source_channel.send(final)

        return trade_data


def is_valid_text(text):
    if text.isalpha():
        return True
    if text.replace(".", "", 1).isdigit() and text.count(".") <= 1:
        return True
    if is_valid_time(text):
        return True
    return False


def is_valid_time(text):
    try:
        if len(text) == 8 and text.count(":") == 2:
            parts = text.split(":")
            if all(part.isdigit() for part in parts):
                hours, minutes, seconds = map(int, parts)
                return 0 <= hours < 24 and 0 <= minutes < 60 and 0 <= seconds < 60
        return False
    except:
        return False

def determine_row_color(original_image, line_text, bounding_data):
    """
    Determines the predominant color (red or green) for a specific row using the original image.
    """
    time_value = line_text.split()[0]  # Extract the time value (e.g., '09:18:06')
    num_boxes = len(bounding_data['text'])

    for i in range(num_boxes):
        detected_text = bounding_data['text'][i].strip()
        if detected_text == time_value:  # Match the time value
            print(f"Matched Time: {detected_text}")

            # Extract bounding box for the time value
            top = bounding_data['top'][i]
            height = bounding_data['height'][i]

            # Define cropping region for the row
            row_region = (
                0,                             # Full width of the image
                max(0, top - 5),               # Top with a small margin
                original_image.shape[1],       # Full width
                min(original_image.shape[0], top + height + 5)  # Bottom with a small margin
            )
            print(f"Row region {row_region}: ")

            # Crop the specific row region from the original image
            row_cropped = original_image[row_region[1]:row_region[3], row_region[0]:row_region[2]]

            # Split the cropped row into color channels
            b, g, r = cv2.split(row_cropped)

            # Calculate average red and green values, ignoring low-intensity (background) values
            avg_red = np.mean(r[r > 50]) if np.any(r > 50) else 0
            avg_green = np.mean(g[g > 50]) if np.any(g > 50) else 0

            print(f"Avg Red: {avg_red:.2f}, Avg Green: {avg_green:.2f}")

            # Determine action based on color
            if avg_green > avg_red:
                return "BUY"
            elif avg_red > avg_green:
                return "SELL"
            else:
                return "UNKNOWN"

    print(f"No matching time value for: {time_value}")
    return "UNKNOWN"


_BORROW_FEE_RE = re.compile(r"(?:^|\s)(?:C\s+)?STOCK BORROW FEE\s+([A-Z0-9.\-]+)\s*$", re.I)
_BORROW_FEE_DESC_RE = re.compile(
    r"(\d{1,2}/\d{1,2})\s+(?:C\s+)?STOCK BORROW FEE\s+([A-Z0-9.\-]+)",
    re.I,
)


def _parse_fee_amount(raw):
    if raw is None:
        return None
    s = str(raw).strip().replace(",", "").replace("$", "")
    if not s:
        return None
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]
    try:
        val = float(s)
    except Exception:
        return None
    return -val if neg else val


def _parse_fee_date(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.split()[0]
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except Exception:
            pass
    return None


def _parse_desc_fee_date(desc, row_date):
    if not desc:
        return None, None
    match = _BORROW_FEE_DESC_RE.search(desc)
    if not match:
        return None, None
    if row_date is None:
        return None, None
    month_day = match.group(1)
    symbol = match.group(2).upper()
    try:
        month, day = [int(part) for part in month_day.split("/")]
        fee_date = datetime(row_date.year, month, day).date()
        if fee_date > row_date:
            fee_date = datetime(row_date.year - 1, month, day).date()
    except Exception:
        return None, symbol
    return fee_date, symbol


def _parse_eod_date(raw):
    try:
        return datetime.strptime(str(raw).strip(), "%m/%d/%y").date()
    except Exception:
        return None


def _parse_borrow_fee_rows(file_path, debug_info=None):
    rows = []
    with open(file_path, "r", encoding="utf-8-sig", errors="ignore", newline="") as f:
        raw_text = f.read()

    if not raw_text.strip():
        return rows

    sample = raw_text[:8192]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",\t;|")
        delimiter = dialect.delimiter
    except csv.Error:
        header_line = sample.splitlines()[0] if sample else ""
        delimiters = [",", "\t", ";", "|"]
        counts = {d: header_line.count(d) for d in delimiters}
        delimiter = max(counts, key=counts.get) if counts and max(counts.values()) > 0 else ","

    lines = [line for line in raw_text.splitlines() if line.strip()]
    if debug_info is not None:
        debug_info["line_count"] = len(lines)
    header_idx = 0
    for i, line in enumerate(lines):
        cols = [c.strip().lower() for c in line.split(delimiter)]
        if "date" in cols and "description" in cols and "amount" in cols:
            header_idx = i
            break

    csv_text = "\n".join(lines[header_idx:])
    reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)

    header_map = {h.strip().lower(): h for h in (reader.fieldnames or []) if h}
    header_keys = list(header_map.keys())
    if debug_info is not None:
        debug_info["delimiter"] = delimiter
        debug_info["header_keys"] = header_keys
        debug_info["header_line"] = lines[header_idx] if lines else ""
        fee_lines = [ln for ln in lines if "STOCK BORROW FEE" in ln.upper()]
        debug_info["fee_line_count"] = len(fee_lines)
        debug_info["fee_line_sample"] = fee_lines[0] if fee_lines else ""

    def get_field(row, *names):
        for name in names:
            key = header_map.get(name)
            if key:
                return row.get(key, "")
            # fallback: partial match (e.g., "description " or "symbol/cusip")
            for cand in header_keys:
                if name in cand:
                    return row.get(header_map[cand], "")
        return ""

    total_rows = 0
    matched_desc = 0
    parsed_dates = 0
    parsed_amounts = 0
    symbol_missing = 0
    sample_fields = None
    sample_match = None

    for row in reader:
        total_rows += 1
        desc = str(get_field(row, "description")).strip()
        desc_norm = " ".join(desc.split())
        if "STOCK BORROW FEE" not in desc_norm.upper():
            continue
        matched_desc += 1
        if sample_match is None:
            sample_match = {
                "date": get_field(row, "date"),
                "description": desc,
                "amount": get_field(row, "amount"),
                "symbol": get_field(row, "symbol", "symbol/cusip"),
            }

        row_date = _parse_fee_date(get_field(row, "date"))
        if not row_date:
            continue
        parsed_dates += 1

        amount = _parse_fee_amount(get_field(row, "amount"))
        if amount is None:
            continue
        parsed_amounts += 1

        desc_date, desc_symbol = _parse_desc_fee_date(desc_norm, row_date)
        fee_date = desc_date or row_date

        symbol = str(get_field(row, "symbol", "symbol/cusip")).strip().upper()
        if not symbol:
            if desc_symbol:
                symbol = desc_symbol
            else:
                match = _BORROW_FEE_RE.search(desc_norm)
                if match:
                    symbol = match.group(1).upper()
                else:
                    last_token = desc_norm.split()[-1] if desc_norm else ""
                    if last_token and SYMB_RE.match(last_token):
                        symbol = last_token

        if not symbol:
            symbol_missing += 1
            continue

        rows.append({
            "symbol": symbol,
            "date": fee_date,
            "amount": amount,
            "description": desc,
            "date_source": "desc" if desc_date else "row",
            "row_date": row_date,
        })

        if sample_fields is None:
            sample_fields = {
                "date": get_field(row, "date"),
                "description": desc,
                "amount": get_field(row, "amount"),
                "symbol": get_field(row, "symbol", "symbol/cusip"),
                "parsed_symbol": symbol,
                "fee_date": fee_date.strftime("%Y-%m-%d"),
            }

    if debug_info is not None:
        debug_info["total_rows"] = total_rows
        debug_info["matched_desc"] = matched_desc
        debug_info["parsed_dates"] = parsed_dates
        debug_info["parsed_amounts"] = parsed_amounts
        debug_info["symbol_missing"] = symbol_missing
        debug_info["sample_fields"] = sample_fields
        debug_info["sample_match"] = sample_match

    return rows


def import_borrow_fees_csv(file_path):
    debug_info = {}
    fee_rows = _parse_borrow_fee_rows(file_path, debug_info=debug_info)
    if not fee_rows:
        debug_bits = [
            f"delimiter={debug_info.get('delimiter', '?')}",
            f"header_keys={debug_info.get('header_keys', [])}",
            f"fee_lines={debug_info.get('fee_line_count', 0)}",
            f"rows={debug_info.get('total_rows', 0)}",
            f"desc_match={debug_info.get('matched_desc', 0)}",
            f"dates={debug_info.get('parsed_dates', 0)}",
            f"amounts={debug_info.get('parsed_amounts', 0)}",
            f"symbol_missing={debug_info.get('symbol_missing', 0)}",
        ]
        sample = debug_info.get("fee_line_sample", "")
        if sample:
            debug_bits.append(f"sample={sample[:200]}")
        sample_fields = debug_info.get("sample_fields")
        if sample_fields:
            debug_bits.append(f"fields={sample_fields}")
        sample_match = debug_info.get("sample_match")
        if sample_match:
            debug_bits.append(f"match={sample_match}")
        return "No borrow fee rows found in the file. Debug: " + " | ".join(debug_bits)

    desc_date_count = sum(1 for r in fee_rows if r.get("date_source") == "desc")
    row_date_count = sum(1 for r in fee_rows if r.get("date_source") == "row")

    min_date = min(r["date"] for r in fee_rows)
    max_date = max(r["date"] for r in fee_rows)

    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()

    before = conn.total_changes
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    insert_rows = []
    delete_rows = []
    for r in fee_rows:
        fee_date_str = r["date"].strftime("%Y-%m-%d")
        insert_rows.append((r["symbol"], fee_date_str, r["amount"], r.get("description", ""), now_ts))
        if r.get("date_source") == "desc":
            row_date = r.get("row_date")
            if row_date and row_date != r["date"]:
                delete_rows.append(
                    (
                        r["symbol"],
                        row_date.strftime("%Y-%m-%d"),
                        r["amount"],
                        r.get("description", ""),
                    )
                )

    if delete_rows:
        cursor.executemany(
            """
            DELETE FROM short_fee_history
            WHERE symbol = ? AND fee_date = ? AND amount = ? AND description = ?
            """,
            delete_rows,
        )
    cursor.executemany(
        """
        INSERT OR IGNORE INTO short_fee_history (symbol, fee_date, amount, description, imported_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        insert_rows,
    )
    inserted = conn.total_changes - before

    # Recompute short_fees from history (deduped)
    cursor.execute("SELECT MAX(fee_date) FROM short_fee_history")
    max_hist_date_raw = cursor.fetchone()[0]
    max_hist_date = datetime.strptime(max_hist_date_raw, "%Y-%m-%d").date() if max_hist_date_raw else None

    cursor.execute("SELECT id, symbol, date_first_open, date_start, date_closed FROM eod_shorts")
    eod_rows = cursor.fetchall()

    updates = []
    eod_symbols = set()
    for row_id, symbol, date_first_open, date_start, date_closed in eod_rows:
        sym = (symbol or "").upper()
        eod_symbols.add(sym)
        start_date = _parse_eod_date(date_first_open) or _parse_eod_date(date_start)
        end_date = _parse_eod_date(date_closed) or max_hist_date
        if not start_date or not end_date:
            updates.append((0.0, row_id))
            continue

        cursor.execute(
            """
            SELECT COALESCE(SUM(amount), 0.0)
            FROM short_fee_history
            WHERE symbol = ? AND fee_date >= ? AND fee_date <= ?
            """,
            (sym, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")),
        )
        sum_amount = cursor.fetchone()[0] or 0.0
        total_fee = -float(sum_amount)
        updates.append((total_fee, row_id))

    if updates:
        cursor.executemany("UPDATE eod_shorts SET short_fees = ? WHERE id = ?", updates)

    conn.commit()
    conn.close()

    missing = sorted(set(r["symbol"] for r in fee_rows) - eod_symbols)
    msg = (
        f"Borrow fees imported: {len(fee_rows)} rows, {len(set(r['symbol'] for r in fee_rows))} symbols. "
        f"Inserted {inserted} new rows. "
        f"Desc dates used: {desc_date_count}, fallback to row date: {row_date_count}. "
        f"Date range: {min_date:%Y-%m-%d} to {max_date:%Y-%m-%d}. "
        f"Updated {len(updates)} cycles."
    )
    if missing:
        preview = ", ".join(missing[:20])
        suffix = "..." if len(missing) > 20 else ""
        msg += f" Unknown symbols: {preview}{suffix}."
    msg += " Open positions use the CSV max date as the end date."
    return msg


def generate_trade_report(trades):
    """
    Generate a fixed-width table for trade summaries using code blocks with emphasized headers.
    Exclude BUY trades unless they reduce an existing short position.
    """
    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()

    # Sort trades by time
    sorted_trades = sorted(trades, key=lambda x: x['time'])

    # Group trades by symbol and action
    combined_trades = {}
    for trade in sorted_trades:
        symbol = trade['symbol']
        action = trade['action']
        price = float(trade['price'])
        qty = int(trade['qty'])

        if action == "BUY":
            # Check if there is an existing short position for the symbol
            cursor.execute('SELECT total_qty FROM trades WHERE symbol = ?', (symbol,))
            row = cursor.fetchone()
            if not row or row[0] >= 0:
                # Skip the BUY if no short position exists
                continue

        # Group by symbol and action
        key = (symbol, action)
        if key not in combined_trades:
            combined_trades[key] = {'symbol': symbol, 'action': action, 'qty': 0, 'price': 0, 'time': trade['time']}
        combined_qty = qty
        if action == "SELL":
            combined_qty *= -1  # Make quantity negative for SELL trades
        combined_trades[key]['qty'] += combined_qty
        combined_trades[key]['price'] = ((combined_trades[key]['price'] * (abs(combined_trades[key]['qty']) - abs(combined_qty))) + (price * abs(combined_qty))) / abs(combined_trades[key]['qty'])

    # Code block table header
    report = "**Trade Summary**\n```\n"
    report += f"{'SYMBOL':<6} {'ACTION':<6} {'PRICE':<7} {'QTY':<5}\n"
    report += f"{'-'*6} {'-'*6} {'-'*7} {'-'*5}\n"

    # Populate table rows
    for trade in combined_trades.values():
        update_trade_db(trade)  # Update the database
        action_display = "Short" if trade['action'] == "SELL" else "Cover"
        report += f"{trade['symbol']:<6} {action_display:<6} {trade['price']:<7.2f} {trade['qty']:<5}\n"

    conn.close()

    # Close code block
    report += "```"
    return report

def _today_mmddyy():
    return datetime.now().strftime('%m/%d/%y')

def _today_date_str():
    # for timestamps in trade_history you already use YYYY-MM-DD HH:MM:SS
    return datetime.now().strftime('%Y-%m-%d')

def _is_eod_candidate(symbol: str) -> bool:
    sym = (symbol or "").upper()

    # 1) in today's in-memory Finviz list?
    if any((s or "").upper() == sym for s in bot_state.finviz_today):
        return True

    # 2) already tracked as an OPEN EOD short in DB (survives restarts)?
    import sqlite3
    conn = sqlite3.connect('trades.db')
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT 1 FROM eod_shorts WHERE symbol = ? AND date_closed IS NULL LIMIT 1",
            (sym,)
        )
        return cur.fetchone() is not None
    finally:
        conn.close()

def update_trade_db(trade):
    """
    Update main trades tables (trades, trade_history) and, if applicable,
    update the EOD shorts table according to the EOD rules.

    EOD rules:
      - Only engage if symbol in bot_state.finviz_today AND action == SELL/BUY (short/cover).
      - On SELL (add/open short): upsert eod_shorts; reset date_start to today; update qty/avg_cost.
      - On BUY (cover): add realized P/L from this cover; update qty; if qty==0 → set date_closed=today.
    """

    # If bot_state.relay_enabled is False, skip updating the database to match your current behavior
    if not bot_state.relay_enabled:
        print(f"Skipping DB update for {trade['symbol']} because relay is OFF.")
        return f"Skipping DB update for {trade['symbol']}."

    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()

    symbol = trade['symbol']
    action = trade['action']  # "SELL" for short, "BUY" for cover, you also map to "Short"/"Cover" in report
    price = float(trade['price'])
    qty = int(trade['qty'])
    time_only = trade.get('time', datetime.now().strftime('%H:%M:%S'))
    trade_time = f"{_today_date_str()} {time_only}"

    # Fetch existing position from main table
    cursor.execute('SELECT total_qty, avg_cost, realized_pl FROM trades WHERE symbol = ?', (symbol,))
    row = cursor.fetchone()

    # Block BUY w/o existing short (same as your previous logic)
    if action == "BUY" and (not row or row[0] >= 0):
        conn.close()
        return f"Skipping BUY trade for {symbol}, no short position to cover."

    # Log to trade_history (no dedupe)
    cursor.execute(
        'INSERT INTO trade_history (symbol, action, qty, price, timestamp) VALUES (?, ?, ?, ?, ?)',
        (symbol, action, qty, price, trade_time)
    )

    # === Main trades table updates (unchanged logic, just clearer) ===
    if row:
        total_qty, avg_cost, realized_pl = row

        if action == "SELL":
            # Increase short position; recompute weighted avg_cost
            new_total_qty = total_qty + qty
            new_avg_cost = ((avg_cost * total_qty) + (price * qty)) / new_total_qty
            cursor.execute('UPDATE trades SET total_qty = ?, avg_cost = ? WHERE symbol = ?',
                           (new_total_qty, new_avg_cost, symbol))
            main_msg = (f"Shorted {qty} shares of {symbol} at ${price:.2f}. "
                        f"Holding {new_total_qty} shares at an average cost of ${new_avg_cost:.2f}.")

            # === EOD logic on SELL ===
            if _is_eod_candidate(symbol):
                # upsert eod_shorts for this open position, reset clock today
                cursor.execute('SELECT id, total_qty, avg_cost, realized_pl, date_start, date_first_open FROM eod_shorts WHERE symbol = ? AND date_closed IS NULL',
                               (symbol,))
                e = cursor.fetchone()
                if e:
                    eid, eqty, eavg, erpl, estart, eopen = e
                    # Track eod qty/avg to match main book on add
                    cursor.execute('UPDATE eod_shorts SET total_qty = ?, avg_cost = ?, date_start = ? WHERE id = ?',
                                   (new_total_qty, new_avg_cost, _today_mmddyy(), eid))
                else:
                    cursor.execute(
                        'INSERT INTO eod_shorts (symbol, total_qty, avg_cost, short_fees, realized_pl, date_first_open, date_start, date_closed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (symbol, new_total_qty, new_avg_cost, 0.0, 0.0, _today_mmddyy(), _today_mmddyy(), None)
                    )

        elif action == "BUY":
            # Cover part/all of the short; realized P/L for the covered shares only
            covered_qty = min(qty, abs(total_qty))
            realized_trade_pl = (avg_cost - price) * covered_qty  # short: sell high, buy low
            new_realized_pl = realized_pl + realized_trade_pl
            new_total_qty = total_qty + covered_qty  # total_qty is negative for short; adding covered_qty moves toward 0

            cursor.execute('UPDATE trades SET total_qty = ?, realized_pl = ? WHERE symbol = ?',
                           (new_total_qty, new_realized_pl, symbol))

            if new_total_qty == 0:
                main_msg = (f"Covered {covered_qty} shares of {symbol} at ${price:.2f}. "
                            f"Realized P/L: ${realized_trade_pl:.2f}. Position fully closed.")
            else:
                main_msg = (f"Covered {covered_qty} shares of {symbol} at ${price:.2f}. "
                            f"Realized P/L: ${realized_trade_pl:.2f}. "
                            f"Holding {new_total_qty} shares at an average cost of ${avg_cost:.2f}.")

            # === EOD logic on BUY ===
            if _is_eod_candidate(symbol):
                # Open EOD row must exist if this was tracked; if not, create it conservatively
                cursor.execute('SELECT id, total_qty, avg_cost, realized_pl, date_first_open FROM eod_shorts WHERE symbol = ? AND date_closed IS NULL',
                               (symbol,))
                e = cursor.fetchone()
                if e:
                    eid, eqty, eavg, erpl, eopen = e
                    # Add the realized P/L for this cover leg into eod_shorts.realized_pl
                    new_erpl = (erpl or 0.0) + realized_trade_pl
                    # After cover, set qty equal to main new_total_qty (kept in sync)
                    if new_total_qty == 0:
                        cursor.execute('UPDATE eod_shorts SET total_qty = ?, realized_pl = ?, date_closed = ? WHERE id = ?',
                                       (0, new_erpl, _today_mmddyy(), eid))
                    else:
                        cursor.execute('UPDATE eod_shorts SET total_qty = ?, realized_pl = ? WHERE id = ?',
                                       (new_total_qty, new_erpl, eid))
                else:
                    # If we covered but somehow there was no open eod row (e.g., bot restarted),
                    # create one to capture realized_pl and close it immediately if qty is now 0.
                    dc = _today_mmddyy() if new_total_qty == 0 else None
                    cursor.execute(
                        'INSERT INTO eod_shorts (symbol, total_qty, avg_cost, short_fees, realized_pl, date_first_open, date_start, date_closed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                        (symbol, new_total_qty, avg_cost, 0.0, realized_trade_pl, _today_mmddyy(), _today_mmddyy(), dc)
                    )

        else:
            main_msg = f"Unsupported action: {action} for {symbol}."

    else:
        # No row in trades yet
        if action == "SELL":
            # First short
            cursor.execute('INSERT INTO trades (symbol, total_qty, avg_cost, realized_pl) VALUES (?, ?, ?, ?)',
                           (symbol, qty, price, 0.0))
            main_msg = f"Shorted {qty} shares of {symbol} at ${price:.2f}. Holding {qty} shares at an average cost of ${price:.2f}."

            # === EOD: if in today list, open eod_shorts with fresh clock ===
            if _is_eod_candidate(symbol):
                cursor.execute(
                    'INSERT INTO eod_shorts (symbol, total_qty, avg_cost, short_fees, realized_pl, date_first_open, date_start, date_closed) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (symbol, qty, price, 0.0, 0.0, _today_mmddyy(), _today_mmddyy(), None)
                )
        else:
            main_msg = f"Error: No position for {symbol} to buy to cover."

    conn.commit()
    conn.close()
    return main_msg


async def relay_to_bound_channels(content):
    """
    Relay the given content to all bound channels, with dynamic embed formatting based on content type.
    """
    # Determine content type and color based on the start of the content
    if content.startswith("**Trade Summary**"):
        content_type = "Trade Summary"
        color = 0x2ECC71  # Green
        content = content.replace("**Trade Summary**", "").strip()  # Remove the redundant title
    elif content.startswith("**Current Positions**"):
        content_type = "Positions"
        color = 0x3498DB  # Blue
        content = content.replace("**Current Positions**", "").strip()
    elif content.startswith("**Realized P/L for"):
        content_type = "Day Summary"
        color = 0xE67E22  # Orange
        # Extract and keep the dynamic date from the title
        title_end_index = content.index("**", len("**Realized P/L for"))  # Find where the title ends
        dynamic_title = content[:title_end_index + 2]  # Capture the full title, including the date
        content = content.replace(dynamic_title, "").strip()  # Remove the title from the content
    else:
        content_type = "Default"
        color = 0x95A5A6  # Gray

    # Create the embed
    embed = discord.Embed(
        title=dynamic_title if content_type == "Day Summary" else content_type,  # Use dynamic title for Day Summary
        description=content,  # Embed description is the cleaned-up content
        color=color,
        timestamp=discord.utils.utcnow()
    )
    embed.set_footer(text="Note: I'm not a financial advisor, and my opinions should not be taken as financial advice.")

    # Fetch all bound channels from the database
    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()
    cursor.execute('SELECT server_id, channel_id FROM bindings')
    rows = cursor.fetchall()
    conn.close()

    # Send the embed to each bound channel
    for server_id, channel_id in rows:
        channel = client.get_channel(int(channel_id))
        if channel:
            try:
                await channel.send(embed=embed)
            except Exception as e:
                print(f"Failed to send message to channel {channel_id} in server {server_id}: {e}")

async def _run_finviz_and_notify():
    """
    Runs finviz screener, parses tickers, stores today's list in memory,
    alerts only in target channel, and checks age of open EOD shorts.
    """
    print("[finviz] Running screener...")

    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, FINVIZ_SCRIPT_PATH,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="ignore")
    except Exception as e:
        print(f"[finviz] ERROR running finviz_screener.py: {e}")
        return

    # Parse: lines like "Tickers: ['FMC', 'APLS']" and "ETF_FLAGS: {'FMC': False, 'APLS': True}"
    tickers = []
    etf_flags = {}
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Tickers:"):
            try:
                tickers = ast.literal_eval(stripped.split("Tickers:", 1)[1].strip())
            except Exception:
                tickers = []
        elif stripped.startswith("ETF_FLAGS:"):
            try:
                etf_flags = ast.literal_eval(stripped.split("ETF_FLAGS:", 1)[1].strip())
            except Exception:
                etf_flags = {}

    # Keep today's Finviz set in memory (no DB watchlist)
    bot_state.finviz_today = [s.strip() for s in tickers if isinstance(s, str)]

    # --- Age checks for open EOD shorts (from eod_shorts) ---
    conn = sqlite3.connect("trades.db")
    cursor = conn.cursor()
    cursor.execute("SELECT symbol, date_first_open, date_start FROM eod_shorts WHERE date_closed IS NULL")
    rows = cursor.fetchall()
    conn.close()

    now_utc = datetime.now(tz=ZoneInfo("UTC"))
    five_day, ten_day = [], []
    for sym, date_first_open, date_start in rows:
        start_for_age = date_first_open or date_start
        age = trading_days_between(start_for_age, now_utc)  # your existing helper
        if age >= 10:
            ten_day.append(f"{sym} ({age}d)")
        elif age >= 5:
            five_day.append(f"{sym} ({age}d)")

    # --- Compose & send alert ONLY to the configured channel ---
    msg = "@here **EOD Screener Picks (-30% perf, price ≥ $5)**\n"
    if bot_state.finviz_today:
        header = f"{'TICKER':<6} {'ETF':<3}"
        dash = f"{'-'*6} {'-'*3}"
        lines = [header, dash]
        for sym in bot_state.finviz_today:
            flag = "Y" if etf_flags.get(sym.upper()) else ""
            lines.append(f"{sym:<6} {flag:<3}")
        msg += "```\n" + "\n".join(lines) + "\n```"
    else:
        msg += "```None today```"

    if five_day:
        msg += "\n⚠️ **5+ trading days open:** " + ", ".join(five_day)
    if ten_day:
        msg += "\n❌ **10+ trading days — close:** " + ", ".join(ten_day)

    channel = client.get_channel(ALERT_CHANNEL_ID)
    if not channel:
        guild = client.get_guild(SERVER_ID)
        if guild:
            channel = guild.get_channel(ALERT_CHANNEL_ID)

    if channel:
        await channel.send(msg)
        print(f"[finviz] Alert sent to {ALERT_CHANNEL_ID}")
    else:
        print("[finviz] ERROR: could not resolve alert channel")


async def finviz_daily_scheduler():
    await client.wait_until_ready()
    while not client.is_closed():
        now_et = datetime.now(tz=EASTERN_TZ)
        next_run = next_trading_run_datetime(now_et)
        sleep_seconds = (next_run - now_et).total_seconds()
        await asyncio.sleep(max(1, sleep_seconds))

        # Weekdays only
        if is_trading_day(datetime.now(tz=EASTERN_TZ)):
            await _run_finviz_and_notify()

        await asyncio.sleep(5)  # cooldown

def is_trading_day(when_dt: datetime) -> bool:
    """Trading day = Mon–Fri (ignore holidays on purpose)."""
    d = when_dt.astimezone(EASTERN_TZ).date()
    return d.weekday() < 5  # 0=Mon … 4=Fri

def next_trading_run_datetime(now_et: datetime) -> datetime:
    """
    Return the next 3:55 PM ET that lands on Mon–Fri.
    If it's already past today's 3:55 PM ET (or today is weekend), advance day-by-day
    until we hit a weekday.
    """
    candidate = datetime.combine(now_et.date(), RUN_AT_ET, tzinfo=EASTERN_TZ)
    if now_et >= candidate or not is_trading_day(candidate):
        candidate += timedelta(days=1)
        while not is_trading_day(candidate):
            candidate += timedelta(days=1)
    return candidate
    
def trading_days_between(date_start_str: str, now_dt: datetime) -> int:
    """
    Count trading days (Mon–Fri) from date_start (MM/DD/YY) up to 'now_dt' (inclusive).
    We subtract 1 so the open day counts as day 0, and day 5 triggers after 5 sessions.
    """
    try:
        start = datetime.strptime(date_start_str, "%m/%d/%y").date()
    except Exception:
        return 0

    end = now_dt.astimezone(EASTERN_TZ).date()
    if end < start:
        return 0

    d = start
    days = 0
    while d <= end:
        if d.weekday() < 5:  # Mon–Fri
            days += 1
        d += timedelta(days=1)

    # Make the open day day-0; e.g., Fri open → the following Fri is age 5
    return max(0, days - 1)
    
def _parse_mdy(s):
    # eod_shorts stores MM/DD/YY (your code’s _today_mmddyy()). Convert to date for comparisons.
    # Returns a date-like string compatible with sqlite date() comparisons.
    # Fallback: return None if format is unexpected.
    try:
        dt = datetime.strptime(s, "%m/%d/%y")
        return dt
    except Exception:
        return None

def _sum_entry_notional_for_cycle(cur, symbol, date_start, date_closed):
    """
    For a closed EOD short, compute the total entry notional (sum of SELL dollars) between
    date_start and date_closed using trade_history. Used to normalize % P/L.
    Returns float or None if we can’t compute confidently.
    """
    # If trade_history.time is ISO like 'YYYY-MM-DD HH:MM:SS', build bounds.
    ds = _parse_mdy(date_start) if date_start else None
    de = _parse_mdy(date_closed) if date_closed else None
    if not (ds and de):
        return None

    ds_iso = ds.strftime("%Y-%m-%d 00:00:00")
    de_iso = de.strftime("%Y-%m-%d 23:59:59")

    try:
        cur.execute("""
            SELECT COALESCE(SUM(price * qty), 0.0)
            FROM trade_history
            WHERE symbol = ?
              AND UPPER(action) = 'SELL'
              AND time >= ? AND time <= ?
        """, (symbol, ds_iso, de_iso))
        val = cur.fetchone()[0]
        return float(val) if val is not None and val > 0 else None
    except Exception:
        return None

async def post_eod_summary(client, also_send_channel_id=None, send_alert=True):
    conn = sqlite3.connect("trades.db")
    cur  = conn.cursor()

    # Open positions (date_closed IS NULL)
    cur.execute("""
        SELECT symbol, total_qty, avg_cost, short_fees, date_first_open, date_start
        FROM eod_shorts
        WHERE date_closed IS NULL
        ORDER BY symbol
    """)
    open_rows = cur.fetchall()

    # Closed positions for metrics
    cur.execute("""
        SELECT symbol, realized_pl, short_fees, date_first_open, date_start, date_closed
        FROM eod_shorts
        WHERE date_closed IS NOT NULL
    """)
    closed_rows = cur.fetchall()

    # Format the open holdings table
    header = f"{'SYMBOL':<7} {'SHARES':>7}   {'AVG COST':>8}   {'FEES':>8}   {'DAYS':>4}"
    dash   = "-" * len(header)
    lines  = [header, dash]

    now_utc = datetime.now(tz=ZoneInfo("UTC"))

    for sym, qty, avg_cost, short_fees, d_open, ds in open_rows:
        # Display DB quantity verbatim (shorts negative, longs positive)
        try:
            qty_display = f"{int(qty)}"
        except Exception:
            qty_display = "0"

        price_str = f"${float(avg_cost):.2f}" if avg_cost is not None else "$0.00"
        fees_str = f"${float(short_fees or 0):.2f}"
        start_for_days = d_open or ds
        lines.append(
            f"{sym:<7} {qty_display:>7}   {price_str:>8}   {fees_str:>8}   {trading_days_between(start_for_days, now_utc):>4}"
        )


    if len(lines) == 2:
        lines.append("(no open EOD shorts)")

    # Metrics on CLOSED rows
    win_bools      = []
    per_pos_pct    = []  # per-position percent returns
    total_realized = 0.0
    total_fees     = 0.0

    for sym, realized_pl, short_fees, d_open, ds, de in closed_rows:
        realized = float(realized_pl or 0.0)
        fees = float(short_fees or 0.0)
        net_realized = realized - fees
        total_realized += net_realized
        total_fees += fees
        win_bools.append(net_realized > 0)

        start_for_cycle = d_open or ds
        entry_notional = _sum_entry_notional_for_cycle(cur, sym, start_for_cycle, de)
        if entry_notional and entry_notional > 0:
            per_pos_pct.append(100.0 * (net_realized / entry_notional))
        # else: leave it out of % average if we can’t determine basis confidently

    n_closed  = len(closed_rows)
    n_wins    = sum(1 for w in win_bools if w)
    win_rate  = (100.0 * n_wins / n_closed) if n_closed else 0.0
    avg_pct   = ((total_realized / (n_closed * 1000.0)) * 100.0) if n_closed else None
    
    # --- NEW: separate winner/loser averages using $1,000 base per closed trade ---
    win_amounts   = []
    loss_amounts  = []
    for _sym, pl, sf, *_r in closed_rows:
        net_pl = float(pl or 0.0) - float(sf or 0.0)
        if net_pl > 0:
            win_amounts.append(net_pl)
        elif net_pl < 0:
            loss_amounts.append(-net_pl)
    n_wins_only   = len(win_amounts)
    n_losses_only = len(loss_amounts)
    avg_profit_pct = ((sum(win_amounts)  / (n_wins_only   * 1000.0)) * 100.0) if n_wins_only   else None
    avg_loss_pct   = ((sum(loss_amounts) / (n_losses_only * 1000.0)) * 100.0) if n_losses_only else None

    # Build message
    table_block = "```\n" + "\n".join(lines) + "\n```"

    # aligned labels so % values line up
    label_w = 18
    metrics_lines = [
        f"Closed trades: {n_closed}",
        f"Win rate: {win_rate:.1f}%",
        f"{'Average % Profit:':<{label_w}} " + (f"{avg_profit_pct:.2f}%" if avg_profit_pct is not None else "n/a"),
        f"{'Average % Loss:':<{label_w}} "   + (f"{avg_loss_pct:.2f}%"   if avg_loss_pct   is not None else "n/a"),
        f"Borrow fees: ${total_fees:,.2f}",
        f"Total Net P/L: ${total_realized:,.2f}",
    ]
    metrics_block = "\n".join(metrics_lines)

    msg = f"**EOD Strategy Summary**\n{table_block}\n{metrics_block}"

    conn.close()

    # Post to alert channel (optional)
    if send_alert:
        channel = client.get_channel(ALERT_CHANNEL_ID)
        if channel is None:
            try:
                channel = await client.fetch_channel(ALERT_CHANNEL_ID)
            except Exception:
                channel = None
        if channel:
            await channel.send(msg)

    # Optionally also post to a secondary channel (e.g., source channel for testing)
    if also_send_channel_id and also_send_channel_id != ALERT_CHANNEL_ID:
        extra = client.get_channel(also_send_channel_id)
        if extra is None:
            try:
                extra = await client.fetch_channel(also_send_channel_id)
            except Exception:
                extra = None
        if extra:
            await extra.send(msg)

def get_all_positions():
    """
    Retrieve all active positions from the database (non-zero shares) and generate a condensed report.
    """
    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()

    # Fetch only positions with non-zero shares
    cursor.execute('SELECT symbol, total_qty, avg_cost FROM trades WHERE total_qty != 0')
    rows = cursor.fetchall()
    conn.close()

    if rows:
        # Create a formatted table
        report = "**Current Positions**\n```\n"
        report += f"{'SYMBOL':<8} {'SHARES':<8} {'AVG COST':<12}\n"
        report += "-" * 28 + "\n"
        for symbol, total_qty, avg_cost in rows:
            report += f"{symbol:<8} {total_qty:<8} ${avg_cost:<12.2f}\n"
        report += "```"
        
        print(f"Generated Positions Report:\n{report}")  # Debugging output
        return report
    else:
        print("No active positions found.")  # Debugging output
        return "No active positions found."


def get_daily_summary(date):
    """
    Generate a summary of realized P/L and % Gain/Loss for trades with non-zero P/L.
    Clean up trades with 0 quantity from both trades and trade_history tables.
    """
    conn = sqlite3.connect('trades.db')
    cursor = conn.cursor()

    print(f"Debug: Querying trades for date: {date}")

    try:
        # Step 1: Fetch symbols with non-zero P/L from the trades table
        cursor.execute('''
            SELECT symbol, realized_pl
            FROM trades
            WHERE realized_pl != 0
        ''')
        trades_with_pl = cursor.fetchall()
        print(f"Debug: Trades with P/L: {trades_with_pl}")

        # Step 2: Filter by symbols traded on the given date
        traded_symbols = set()
        for symbol, _ in trades_with_pl:
            cursor.execute('''
                SELECT 1
                FROM trade_history
                WHERE symbol = ? AND DATE(timestamp) = ?
                LIMIT 1
            ''', (symbol, date))
            if cursor.fetchone():
                traded_symbols.add(symbol)

        # Step 3: Generate report for symbols traded on the given date
        report = f"**Realized P/L for {date}**\n```\n"
        report += f"{'SYMBOL':<6} {'REALIZED':<10} {'GAIN/LOSS':<9}\n"
        report += f"{'-' * 6} {'-' * 10} {'-' * 9}\n"

        total_pl = 0
        for symbol, realized_pl in trades_with_pl:
            if symbol not in traded_symbols:
                continue

            # Calculate total sell value from trade history (all-time)
            cursor.execute('''
                SELECT SUM(ABS(qty * price))
                FROM trade_history
                WHERE symbol = ? AND action = "SELL"
            ''', (symbol,))
            total_sell_value = cursor.fetchone()[0] or 0  # Default to 0 if no sell value found

            if total_sell_value > 0:
                # Calculate % Gain/Loss based on total sell value
                percent_gain = (realized_pl / total_sell_value) * 100
            else:
                percent_gain = 0  # No sell value means no % gain/loss calculation

            # Add the symbol to the report
            report += f"{symbol:<6} ${realized_pl:<10.2f} {percent_gain:>7.2f}%\n"
            total_pl += realized_pl

        report += f"{'-' * 6} {'-' * 10} {'-' * 9}\n"
        report += f"{'Total':<6} ${total_pl:<10.2f}\n"
        report += "```"
        print(report)

        # Step 4: Clean up closed trades
        cursor.execute('SELECT symbol FROM trades WHERE total_qty = 0')
        closed_symbols = [row[0] for row in cursor.fetchall()]
        print(f"Debug: Closed symbols: {closed_symbols}")

        for symbol in closed_symbols:
            cursor.execute('DELETE FROM trades WHERE symbol = ?', (symbol,))
            cursor.execute('DELETE FROM trade_history WHERE symbol = ?', (symbol,))

        conn.commit()
        conn.close()
        return report

    except Exception as e:
        print(f"Error during query: {e}")
        conn.close()
        raise

        
initialize_db()
bot_state = BotState()
client.run(DISCORD_TOKEN)
