from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Set
import uuid


@dataclass
class NodeEagerBroadcast:
    """
    Eager-broadcasting multi-leader node.

    - Any node can accept client writes.
    - Every message (client or replication) is forwarded to all peers exactly once
      on the first time it is seen at this node.
    - A per-node `seen_message_ids` cache prevents infinite rebroadcast loops.
    - Delivery is last-writer-wins by arrival order, matching the baseline semantics.
    """

    node_id: str
    peers: List[str]
    transport: Any
    scheduler: Any
    store: Dict[str, Any] = field(default_factory=dict)
    log: List[Tuple[str, Any]] = field(default_factory=list)
    seen_message_ids: Set[str] = field(default_factory=set)

    def brief_state(self):
        return {"id": self.node_id, "kv": dict(self.store)}

    # Client-facing APIs
    def client_put(self, key, value):
        # Create a globally unique id for this client write
        msg_id = str(uuid.uuid4())
        self._receive_and_flood(key, value, msg_id, origin=self.node_id)

    def client_get(self, key):
        return self.store.get(key)

    # Core logic
    def _receive_and_flood(self, key, value, msg_id: str, origin: str):
        first_time = self._mark_seen(msg_id)
        # Always deliver on receive
        self._deliver(key, value)
        # Flood to all peers only if first time seen here
        if first_time:
            self._broadcast_replication(key, value, msg_id, origin)

    def _broadcast_replication(self, key, value, msg_id: str, origin: str):
        for peer in self.peers:
            if peer != self.node_id:
                self.transport.send(
                    peer,
                    {
                        "type": "replicate",
                        "from": self.node_id,
                        "origin": origin,
                        "msg_id": msg_id,
                        "key": key,
                        "value": value,
                    },
                )

    def _mark_seen(self, msg_id: str) -> bool:
        if msg_id in self.seen_message_ids:
            return False
        self.seen_message_ids.add(msg_id)
        return True

    def _deliver(self, key, value):
        self.store[key] = value
        self.log.append((key, value))

    # Network handler
    def on_message(self, msg):
        typ = msg.get("type")
        if typ == "replicate":
            self._receive_and_flood(
                msg["key"], msg["value"], msg["msg_id"], origin=msg.get("origin", "?")
            )
        else:
            raise ValueError(f"Unknown type in message {msg!r}")
            