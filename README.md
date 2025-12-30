# WallStreet

WallStreet is a Discord bot that logs trades, tracks short positions, and posts EOD screeners and summaries.

## Setup
1. Create a `.env` file using `.env.example` as a template.
2. Install dependencies (discord.py, pytesseract, opencv-python, pillow, python-decouple, numpy, finvizfinance, bs4, lxml).
3. Run `0_start_bot.bat`.

## Notes
- The Finviz screener script path is configured in `Discord_bot.py`.
- The bot uses `trades.db` for positions and history.