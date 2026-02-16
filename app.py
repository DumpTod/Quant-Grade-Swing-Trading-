# ============================================================
# app.py — Quant Scanner API Server (Lightweight)
# ============================================================
# This server ONLY serves scan results uploaded from Colab.
# The heavy scanning is done in Google Colab, results are
# posted here via /api/upload endpoint.
# ============================================================

import os
import json
from datetime import datetime
from flask import Flask, jsonify, request
from flask_cors import CORS
from threading import Lock

app = Flask(__name__)
CORS(app)

# ============================================================
# STORAGE — Scan results stored in memory + file backup
# ============================================================
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
scan_lock = Lock()

# Load from file if exists (persists across restarts)
def load_saved_results():
    global SCAN_RESULTS
    try:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, 'r') as f:
                SCAN_RESULTS = json.load(f)
                print(f"✅ Loaded saved results: {SCAN_RESULTS['scan_metadata'].get('timestamp', 'unknown')}")
    except Exception as e:
        print(f"⚠️ Could not load saved results: {e}")

def save_results():
    try:
        with open(DATA_FILE, 'w') as f:
            json.dump(SCAN_RESULTS, f)
    except Exception as e:
        print(f"⚠️ Could not save results: {e}")

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/')
def home():
    return jsonify({
        'name': 'Quant Scanner API',
        'version': '2.0',
        'architecture': 'Colab (scanner) + Render (API server)',
        'endpoints': {
            '/api/scan': 'GET — Returns latest scan results',
            '/api/status': 'GET — Health check',
            '/api/upload': 'POST — Upload scan results from Colab',
            '/api/stock/<SYMBOL>': 'GET — Get specific stock signal'
        }
    })

@app.route('/api/scan', methods=['GET'])
def api_scan():
    """Returns the latest scan results"""
    return jsonify(SCAN_RESULTS)

@app.route('/api/status', methods=['GET'])
def api_status():
    """Health check"""
    meta = SCAN_RESULTS.get('scan_metadata', {})
    return jsonify({
        'status': 'running',
        'last_scan': meta.get('timestamp', 'Never'),
        'buy_signals': meta.get('buy_signals_count', 0),
        'sell_signals': meta.get('sell_signals_count', 0),
        'market_regime': meta.get('market_regime', 'unknown'),
        'stocks_scanned': meta.get('stocks_scanned', 0)
    })

@app.route('/api/upload', methods=['POST'])
def api_upload():
    """
    Upload scan results from Google Colab.

    Colab runs the full scanner, then POSTs results here.
    This endpoint receives and stores them.

    Headers: Content-Type: application/json
    Body: Full scan results JSON
    """
    global SCAN_RESULTS

    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        # Validate structure
        if 'buy_signals' not in data and 'sell_signals' not in data:
            return jsonify({'error': 'Invalid data format — need buy_signals and/or sell_signals'}), 400

        # Store results
        with scan_lock:
            SCAN_RESULTS = {
                'buy_signals': data.get('buy_signals', []),
                'sell_signals': data.get('sell_signals', []),
                'scan_metadata': data.get('scan_metadata', {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'stocks_scanned': 0,
                    'buy_signals_count': len(data.get('buy_signals', [])),
                    'sell_signals_count': len(data.get('sell_signals', [])),
                    'market_regime': 'unknown'
                })
            }

            # Ensure counts are updated
            SCAN_RESULTS['scan_metadata']['buy_signals_count'] = len(SCAN_RESULTS['buy_signals'])
            SCAN_RESULTS['scan_metadata']['sell_signals_count'] = len(SCAN_RESULTS['sell_signals'])

            # Save to file for persistence
            save_results()

        buy_count = len(SCAN_RESULTS['buy_signals'])
        sell_count = len(SCAN_RESULTS['sell_signals'])
        regime = SCAN_RESULTS['scan_metadata'].get('market_regime', 'unknown')

        print(f"✅ Scan results uploaded: {buy_count} buys, {sell_count} sells, regime: {regime}")

        return jsonify({
            'status': 'success',
            'message': f'Uploaded {buy_count} buy signals and {sell_count} sell signals',
            'buy_count': buy_count,
            'sell_count': sell_count,
            'timestamp': SCAN_RESULTS['scan_metadata'].get('timestamp')
        })

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/stock/<symbol>', methods=['GET'])
def api_stock(symbol):
    """Get signal for a specific stock"""
    symbol = symbol.upper()
    for s in SCAN_RESULTS.get('buy_signals', []) + SCAN_RESULTS.get('sell_signals', []):
        if s['symbol'] == symbol:
            return jsonify(s)
    return jsonify({'error': f'{symbol} not found in latest scan'}), 404

@app.route('/api/clear', methods=['POST'])
def api_clear():
    """Clear all stored results"""
    global SCAN_RESULTS
    with scan_lock:
        SCAN_RESULTS = {
            'buy_signals': [],
            'sell_signals': [],
            'scan_metadata': {'timestamp': None, 'stocks_scanned': 0,
                              'buy_signals_count': 0, 'sell_signals_count': 0,
                              'market_regime': 'unknown'}
        }
        save_results()
    return jsonify({'status': 'cleared'})

# ============================================================
# STARTUP
# ============================================================
load_saved_results()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Quant Scanner API starting on port {port}")
    print(f"   Architecture: Colab (scanner) → Render (API server)")
    print(f"   Upload endpoint: POST /api/upload")
    app.run(host='0.0.0.0', port=port)
