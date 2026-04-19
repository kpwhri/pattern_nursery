import hashlib


def stable_shard_id(doc_id: str, total_shards: int) -> int:
    """Compute a stable shard ID for a given document ID."""
    if total_shards <= 1:
        return 0
    # use sha256 for stable hashing across platforms/sessions
    h = hashlib.sha256(str(doc_id).encode('utf8')).digest()
    val = int.from_bytes(h[:8], byteorder='big')
    return val % total_shards
