import pickle
import redis
from typing import Optional, Any, Iterator, Tuple, Sequence
from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint, CheckpointMetadata, CheckpointTuple
from app.config import settings


class RedisSaver(BaseCheckpointSaver):
    """
    Redis-based checkpoint saver for LangGraph.
    Implements the full BaseCheckpointSaver interface.
    """

    def __init__(self, redis_url: Optional[str] = None, prefix: str = 'helpdesk:checkpoint:'):
        super().__init__()
        self.redis_url = redis_url or settings.REDIS_URL
        self.prefix = prefix
        self.client = redis.from_url(self.redis_url, decode_responses=False)

    def _make_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> str:
        return f"{self.prefix}{checkpoint_ns}:{thread_id}:{checkpoint_id}"

    def _make_version_key(self, thread_id: str, checkpoint_ns: str) -> str:
        return f"{self.prefix}version:{checkpoint_ns}:{thread_id}"
    
    def _make_writes_key(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str, task_id: str) -> str:
        return f"{self.prefix}writes:{checkpoint_ns}:{thread_id}:{checkpoint_id}:{task_id}"

    def put(
        self,
        config: dict,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: dict,
    ) -> dict:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")
        checkpoint_id = checkpoint["id"]

        key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)

        payload = {
            "checkpoint": checkpoint,
            "metadata": metadata,
            "new_versions": new_versions,
        }

        self.client.set(key, pickle.dumps(payload))

        # Update "latest" pointer
        latest_key = self._make_key(thread_id, checkpoint_ns, "latest")
        self.client.set(latest_key, checkpoint_id.encode())

        return config

    def put_writes(self, config: dict, writes: Sequence[Tuple[str, Any]], task_id: str) -> None:
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")
        checkpoint_id = config["configurable"].get("checkpoint_id", "pending")

        writes_key = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id, task_id)

        # Convert (channel, value) → (task_id, channel, value)
        triple_writes = [(task_id, w[0], w[1]) for w in writes]

        writes_data = {
            "writes": triple_writes,
            "task_id": task_id,
        }

        self.client.set(writes_key, pickle.dumps(writes_data))


    def get_tuple(self, config: dict) -> Optional[CheckpointTuple]:
        """
        Returns a CheckpointTuple which LangGraph expects.
        CheckpointTuple is a NamedTuple with: (config, checkpoint, metadata, parent_config, pending_writes)
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")
        checkpoint_id = config["configurable"].get("checkpoint_id")

        if not checkpoint_id:
            return None 

        key = self._make_key(thread_id, checkpoint_ns, checkpoint_id)
        payload = self.client.get(key)

        if not payload:
            return None

        try:
            data = pickle.loads(payload)
            
            # Get any pending writes for this checkpoint
            pending_writes = self._get_pending_writes(thread_id, checkpoint_ns, checkpoint_id)
            
            # Return a CheckpointTuple instead of a plain tuple
            return CheckpointTuple(
                config=config,
                checkpoint=data["checkpoint"],
                metadata=data.get("metadata", {}),
                parent_config=None,  # Set this if you track parent checkpoints
                pending_writes=pending_writes
            )
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return None

    def _get_pending_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> list:
        """Get all pending writes for a checkpoint."""
        pattern = self._make_writes_key(thread_id, checkpoint_ns, checkpoint_id, "*")
        pending = []
        
        for key in self.client.scan_iter(match=pattern):
            writes_data_bytes = self.client.get(key)
            if writes_data_bytes:
                try:
                    writes_data = pickle.loads(writes_data_bytes)
                    pending.extend(writes_data.get("writes", []))
                except Exception as e:
                    print(f"Error loading pending writes from {key}: {e}")
                    continue
        
        return pending

    def list(self, config: dict, *, filter: Optional[dict] = None, before: Optional[dict] = None, limit: Optional[int] = None) -> Iterator[CheckpointTuple]:
        """
        List checkpoints matching the given criteria.
        Returns an iterator of CheckpointTuple objects.
        """
        thread_id = config["configurable"]["thread_id"]
        checkpoint_ns = config["configurable"].get("checkpoint_ns", "default")

        pattern = f"{self.prefix}{checkpoint_ns}:{thread_id}:*"
        count = 0

        for key in self.client.scan_iter(match=pattern):
            if limit and count >= limit:
                break
                
            key_str = key.decode() if isinstance(key, bytes) else key
            if key_str.endswith(":latest") or ":version:" in key_str or ":writes:" in key_str:
                continue

            payload = self.client.get(key)
            if payload:
                try:
                    data = pickle.loads(payload)
                    
                    # Extract checkpoint_id from key
                    checkpoint_id = key_str.split(":")[-1]
                    pending_writes = self._get_pending_writes(thread_id, checkpoint_ns, checkpoint_id)
                    
                    yield CheckpointTuple(
                        config=config,
                        checkpoint=data["checkpoint"],
                        metadata=data.get("metadata", {}),
                        parent_config=None,
                        pending_writes=pending_writes
                    )
                    count += 1
                except Exception as e:
                    print(f"Error loading checkpoint from {key_str}: {e}")
                    continue

    def get_next_version(self, current: Optional[int], channel: str) -> int:
        return 1 if current is None else current + 1


# Instantiate default checkpointer
checkpointer = RedisSaver()