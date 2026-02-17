# ============================================================
# app.py — Quant Scanner API (Lightweight Server)
# ============================================================
# Receives results from Colab, serves them to the webpage.
# No scanning on Render — yfinance is blocked on datacenter IPs.
# ============================================================

import os
import json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Lock

app = Flask(__name__)
CORS(app)

SCAN_RESULTS = {
    'buy_signals': [],
    'sell_signals': [],
    'scan_metadata': {
        'timestamp': None,
        'stocks_scanned': 0,
        'buy_signals_count': 0,
        'sell_signals_count': 0,
        'market_regime': 'unknown',
        'regime_probs': {},
        'scan_duration_seconds': 0,
        'failed_stocks': 0
    }
}

DATA_FILE = 'scan_results.json'
lock = Lock()

def load_saved():
    global SCAN_RESULTS
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                SCAN_RESULTS = json.load(f)
                ts = SCAN_RESULTS.get('scan_metadata', {}).get('timestamp', 'unknown')
                print(f"Loaded saved results from {ts}")
    except Exception as e:
        print(f"No saved results: {e}")

def save_results():
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(SCAN_RESULTS, f)
    except Exception as e:
        print(f"Save failed: {e}")

@app.route('/')
def home():
    return jsonify({
        'name': 'Quant Scanner API',
        'version': '2.0',
        'status': 'running',
        'architecture': 'Colab scans → Render serves',
        'endpoints': {
            '/api/scan': 'GET latest results',
            '/api/status': 'Health check',
            '/api/upload': 'POST results from Colab',
            '/api/stock/<SYMBOL>': 'GET single stock'
        }
    })

@app.route('/api/scan', methods=['GET'])
def api_scan():
    return jsonify(SCAN_RESULTS)

@app.route('/api/status', methods=['GET'])
def api_status():
    m = SCAN_RESULTS.get('scan_metadata', {})
    return jsonify({
        'status': 'running',
        'last_scan': m.get('timestamp', 'Never'),
        'buy_signals': m.get('buy_signals_count', 0),
        'sell_signals': m.get('sell_signals_count', 0),
        'market_regime': m.get('market_regime', 'unknown'),
        'stocks_scanned': m.get('stocks_scanned', 0)
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    global SCAN_RESULTS
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        with lock:
            SCAN_RESULTS = {
                'buy_signals': data.get('buy_signals', []),
                'sell_signals': data.get('sell_signals', []),
                'scan_metadata': data.get('scan_metadata', {})
            }
            SCAN_RESULTS['scan_metadata']['buy_signals_count'] = len(SCAN_RESULTS['buy_signals'])
            SCAN_RESULTS['scan_metadata']['sell_signals_count'] = len(SCAN_RESULTS['sell_signals'])
            save_results()

        b = len(SCAN_RESULTS['buy_signals'])
        s = len(SCAN_RESULTS['sell_signals'])
        print(f"Upload OK: {b} buys, {s} sells")

        return jsonify({'status': 'success', 'buy_count': b, 'sell_count': s})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def api_stock(symbol):
    symbol = symbol.upper()
    for s in SCAN_RESULTS.get('buy_signals', []) + SCAN_RESULTS.get('sell_signals', []):
        if s['symbol'] == symbol:
            return jsonify(s)
    return jsonify({'error': f'{symbol} not found'}), 404

load_saved()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
