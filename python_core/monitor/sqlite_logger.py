import sqlite3
import time
from functools import wraps
import json

from typing import Any, Callable

def init_db() -> None:
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

def instrument_inference(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator for standardized observability.
    Logs every inference request to a SQLite database for the Measurement Study.
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        start: float = time.perf_counter()
        
        # Execute the original inference call
        response: Any = await func(*args, **kwargs)
        
        latency: float = (time.perf_counter() - start) * 1000
        request_obj = args[0] if args else (kwargs.get("req") or kwargs.get("request") or (list(kwargs.values())[0] if kwargs else None))
        prompt_len: int = len(request_obj.prompt) if request_obj and hasattr(request_obj, "prompt") else 0
        
        engine: str = response.engine_id if hasattr(response, "engine_id") else "unknown"
        status: int = 200 if getattr(response, "success", False) else 500
        cost: float = getattr(response, "cost_usd", 0.0)
        was_escalated: bool = getattr(response, "was_escalated", False)
        
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

