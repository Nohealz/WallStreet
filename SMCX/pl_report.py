import argparse
import os
import sys
import zipfile
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP, getcontext
from datetime import datetime, timedelta

getcontext().prec = 28

NS = {'main': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
REL_NS = {'rel': 'http://schemas.openxmlformats.org/package/2006/relationships'}

FEE_COLUMNS = [
    'TransFee',
    'ECNTaker',
    'ECNMaker',
    'ORFFee',
    'TAFFee',
    'SECFee',
    'CATFee',
    'Commissions',
]


def d(value):
    if value is None:
        return Decimal('0')
    text = str(value).strip()
    if text == '':
        return Decimal('0')
    return Decimal(text)


def fmt2(value):
    if not isinstance(value, Decimal):
        value = d(value)
    return f"{value.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}"


def read_xlsx_rows(path, sheet_name='Sheet1'):
    """Minimal XLSX reader for this file format (no external deps)."""
    with zipfile.ZipFile(path) as z:
        shared = []
        try:
            with z.open('xl/sharedStrings.xml') as f:
                tree = ET.parse(f)
                for si in tree.getroot().findall('main:si', NS):
                    texts = [t.text or '' for t in si.findall('.//main:t', NS)]
                    shared.append(''.join(texts))
        except KeyError:
            pass

        # Map sheet name -> sheet file
        with z.open('xl/workbook.xml') as f:
            wb = ET.parse(f)
            sheets = wb.getroot().find('main:sheets', NS)
            sheet_map = {}
            for s in sheets.findall('main:sheet', NS):
                name = s.get('name')
                rid = s.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id')
                sheet_map[name] = rid

        if sheet_name not in sheet_map:
            raise ValueError(f"Sheet '{sheet_name}' not found. Available: {', '.join(sheet_map.keys())}")

        # relationships map
        with z.open('xl/_rels/workbook.xml.rels') as f:
            rels = ET.parse(f)
            rid_to_target = {}
            for r in rels.getroot().findall('rel:Relationship', REL_NS):
                rid_to_target[r.get('Id')] = r.get('Target')

        target = rid_to_target[sheet_map[sheet_name]]
        sheet_path = f"xl/{target}"

        with z.open(sheet_path) as f:
            sheet = ET.parse(f).getroot()

        rows = []
        for row in sheet.findall('.//main:sheetData/main:row', NS):
            r = []
            for c in row.findall('main:c', NS):
                v = c.find('main:v', NS)
                if v is None:
                    val = ''
                else:
                    val = v.text or ''
                if c.get('t') == 's':
                    try:
                        val = shared[int(val)]
                    except Exception:
                        pass
                r.append(val)
            rows.append(r)
        return rows


def excel_date_to_iso(excel_serial):
    # Excel 1900 date system (with the 1900 leap year bug)
    base = datetime(1899, 12, 30)
    return (base + timedelta(days=float(excel_serial))).date().isoformat()


def extract_trades(rows, symbol_filter=None):
    headers = rows[0]
    idx = {h: i for i, h in enumerate(headers)}

    required = ['Date', 'Quantity', 'Symbol', 'Price']
    for col in required:
        if col not in idx:
            raise ValueError(f"Missing required column: {col}")

    trades = []
    for i, r in enumerate(rows[1:], start=1):
        symbol = r[idx['Symbol']] if idx['Symbol'] < len(r) else ''
        if symbol_filter and symbol != symbol_filter:
            continue

        qty = d(r[idx['Quantity']])
        price = d(r[idx['Price']])
        date_raw = r[idx['Date']] if idx['Date'] < len(r) else ''
        try:
            date_val = float(date_raw)
        except Exception:
            date_val = 0.0

        fees = Decimal('0')
        for col in FEE_COLUMNS:
            if col in idx and idx[col] < len(r):
                fees += d(r[idx[col]])

        trades.append({
            'row': i,
            'date_serial': date_val,
            'date_iso': excel_date_to_iso(date_val) if date_val else '',
            'qty': qty,
            'price': price,
            'fees': fees,
        })
    return trades


def average_pl(trades, last_price):
    buy_qty = Decimal('0')
    buy_value = Decimal('0')
    sell_qty = Decimal('0')
    sell_value = Decimal('0')

    for t in trades:
        qty = t['qty']
        price = t['price']
        fees = t['fees']
        if qty == 0:
            continue

        if qty > 0:
            buy_qty += qty
            buy_value += qty * price + fees
        else:
            sqty = -qty
            sell_qty += sqty
            sell_value += sqty * price - fees

    avg_buy = (buy_value / buy_qty) if buy_qty else Decimal('0')
    avg_sell = (sell_value / sell_qty) if sell_qty else Decimal('0')

    closed_qty = min(buy_qty, sell_qty)
    realized = (avg_sell - avg_buy) * closed_qty

    open_qty = buy_qty - sell_qty
    if open_qty > 0:
        unrealized = (last_price - avg_buy) * open_qty
    elif open_qty < 0:
        unrealized = (avg_sell - last_price) * (-open_qty)
    else:
        unrealized = Decimal('0')

    return {
        'position_qty': open_qty,
        'realized': realized,
        'unrealized': unrealized,
        'avg_buy': avg_buy,
        'avg_sell': avg_sell,
        'buy_qty': buy_qty,
        'sell_qty': sell_qty,
    }


def main():
    p = argparse.ArgumentParser(description='Compute realized/unrealized P/L from account activity XLSX.')
    p.add_argument('--file', help='Path to XLS/XLSX export file')
    p.add_argument('--symbol', required=True, help='Symbol to analyze')
    p.add_argument('--sheet', default='Sheet1', help='Sheet name (default: Sheet1)')
    p.add_argument('--price', help='Manual last price (overrides --price-url)')

    args = p.parse_args()

    file_path = args.file
    if not file_path:
        candidates = [f for f in os.listdir('.') if f.lower().endswith(('.xls', '.xlsx'))]
        if not candidates:
            raise SystemExit('No .xls/.xlsx files found in the current folder.')

        print('Select a file:')
        for i, name in enumerate(candidates, start=1):
            print(f'  {i}. {name}')
        while True:
            choice = input('Enter file number: ').strip()
            try:
                idx = int(choice)
                if 1 <= idx <= len(candidates):
                    file_path = candidates[idx - 1]
                    break
            except Exception:
                pass
            print('Invalid choice. Try again.', file=sys.stderr)

    rows = read_xlsx_rows(file_path, sheet_name=args.sheet)
    trades = extract_trades(rows, symbol_filter=args.symbol)
    if not trades:
        raise SystemExit(f'No trades found for symbol {args.symbol}')

    if args.price is not None:
        last_price = d(args.price)
        price_source = 'manual'
    else:
        while True:
            raw = input(f'Enter current price for {args.symbol}: ').strip()
            if raw:
                try:
                    last_price = d(raw)
                    price_source = 'prompt'
                    break
                except Exception:
                    pass
            print('Invalid price. Try again.', file=sys.stderr)

    result = average_pl(trades, last_price)

    print(f"Symbol: {args.symbol}")
    print(f"Last price: {fmt2(last_price)} ({price_source})")
    print(f"Position qty: {fmt2(result['position_qty'])}")
    print(f"Buy qty: {fmt2(result['buy_qty'])} @ avg price {fmt2(result['avg_buy'])}")
    print(f"Sell qty: {fmt2(result['sell_qty'])} @ avg price {fmt2(result['avg_sell'])}")
    print(f"Realized P/L (Average): {fmt2(result['realized'])}")
    print(f"Unrealized P/L: {fmt2(result['unrealized'])}")


if __name__ == '__main__':
    main()
