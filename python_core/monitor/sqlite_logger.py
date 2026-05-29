import sqlite3
import time
from functools import wraps
import json

def init_db():
    conn = sqlite3.connect('inference_research.db')
    conn.execute('''
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_length INTEGER,
            engine_selected TEXT,
            status_code INTEGER,
            latency_ms REAL,
            estimated_cost_usd REAL,
            quality_score REAL,
            was_escalated BOOLEAN
        )
    ''')
    conn.close()

init_db()

def instrument_inference(func):
    """
    Decorator for standardized observability.
    Logs every inference request to a SQLite database for the Measurement Study.
    """
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start = time.perf_counter()
        
        # Execute the original inference call
        response = await func(request, *args, **kwargs)
        
        latency = (time.perf_counter() - start) * 1000
        prompt_len = len(request.prompt) if hasattr(request, "prompt") else 0
        
        engine = response.engine_id if hasattr(response, "engine_id") else "unknown"
        status = 200 if getattr(response, "success", False) else 500
        cost = getattr(response, "cost_usd", 0.0)
        was_escalated = getattr(response, "was_escalated", False)
        
        try:
            conn = sqlite3.connect('inference_research.db')
            conn.execute('''
                INSERT INTO inferences (prompt_length, engine_selected, status_code, latency_ms, estimated_cost_usd, quality_score, was_escalated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (prompt_len, engine, status, latency, cost, None, was_escalated))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Failed to log to SQLite: {e}")
            
        return response
    return wrapper
