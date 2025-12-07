import hashlib
import pickle
import redis
from app.config import settings

redis_client = redis.from_url(settings.REDIS_URL, decode_responses=False)

# ✅ CACHE KEY NOW DEPENDS ONLY ON QUERY (NO INTENT)
def _key(query: str):
    h = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"helpdesk:cache:{h}"

# ✅ UPDATED SIGNATURE (NO INTENT)
def get_cached(query: str):
    val = redis_client.get(_key(query))
    if not val:
        return None
    try:
        return pickle.loads(val)
    except Exception:
        return None

# ✅ UPDATED SIGNATURE (NO INTENT)
def set_cached(query: str, value, ttl: int = 3600):
    try:
        redis_client.set(_key(query), pickle.dumps(value), ex=ttl)
    except Exception as e:
        print("Cache set failed", e)
