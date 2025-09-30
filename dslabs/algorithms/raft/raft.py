"""Guidance and scaffolding for implementing the Raft consensus algorithm.

This module mirrors the structure described in *Raft: In Search of an
Understandable Consensus Algorithm* (Ongaro & Ousterhout, 2014). The goal is to
provide you with explicit hooks, rich documentation, and light-weight
scaffolding so that the implementation can focus on the algorithmic ideas:
leader election, log replication, safety, and the interaction with application
state machines.

You should:

* Read the Raft paper and map each major concept to the methods declared here.
* Rely on `dslabs.protocols.Transport` for network I/O and `dslabs.protocols.Scheduler`
  for timers; the unit tests in tests/test_raft_algorithm.py inject fake
  implementations of these protocols to keep the logic deterministic.
* Follow the provided docstrings and inline comments as a step-by-step outline
  when filling in each method. The comments are not exhaustive, but they call
  out important conditions, state transitions, and message flows that must be handled.

Until the algorithm is implemented, the stubs intentionally raise `NotImplementedError`
so that the unit tests fail, reminding you to finish the implementation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from dslabs.protocols import Scheduler, SchedulerCancel, Transport


class RaftState(str, Enum):
    """
    High-level role assumed by a Raft node.

    Raft rotates between three roles:

    ``FOLLOWER``
        Passive role, responds to requests from leaders or candidates and resets
        its election timeout when heartbeats arrive.

    ``CANDIDATE``
        Initiated after an election timeout; the node increments its term,
        votes for itself, and requests votes from peers in pursuit of
        leadership.

    ``LEADER``
        The node responsible for log replication and serving client requests.
        Leaders send periodic AppendEntries heartbeats to maintain authority.
    """

    FOLLOWER = "follower"
    CANDIDATE = "candidate"
    LEADER = "leader"


@dataclass
class LogEntry:
    """
    Single entry stored within the replicated log.

    Parameters
    ----------
    term:
        Election term under which the entry was created by the leader. Raft
        relies on term comparisons to uphold the *log matching* property.
    command:
        Opaque client command that should be applied to the state machine once
        committed. The implementation decides the structure (dict, tuple, etc.).
    """

    term: int
    command: Any


@dataclass
class Raft:
    """
    Skeleton of the Raft state machine.

    Parameters:
    ----------
    node_id:
        Identifier for this node. Raft messages carry string identifiers and
        the tests use human-readable values ("n1", "n2", etc.).
    peers:
        List of node identifiers participating in the cluster. The local node
        may or may not appear in this list, depending on how the runner builds
        membership; the implementation should be robust to either choice.
    transport:
        Implementation of `dslabs.protocols.Transport` used to send Raft
        protocol messages. You must call `transport.send` with JSON-like
        dictionaries describing RequestVote and AppendEntries interactions.
    scheduler:
        Implementation of `dslabs.protocols.Scheduler` that provides election
        timeouts and heartbeat intervals. Timers are critical to Raft’s
        liveness guarantees.
    apply:
        Callback invoked with ``(command, index)`` whenever an entry becomes
        committed and should be applied to the replicated state machine. The
        tests assert on this hook to check that commits are signalled correctly.

    Attributes
    ----------
    state:
        Current `RaftState`. Start as follower, transition per the paper.
    current_term / voted_for:
        Persistent election metadata. `current_term` increments on new elections;
        `voted_for` tracks which candidate received our vote in the current term
        (or `None` if no vote was cast yet).
    log:
        In-memory log containing `LogEntry` entries. Index 0 corresponds to the
        first command appended by any leader.
    commit_index / last_applied:
        Match the definitions from the paper. `commit_index` tracks the highest
        log index known to be committed; `last_applied` is the highest index
        already delivered to `apply`.
    next_index / match_index:
        Leader-only replication metadata. `next_index` is the next log index
        that should be sent to each follower; `match_index` stores the highest
        index known to be replicated on each follower.
    leader_id:
        Convenience field to remember the current leader (useful for followers
        redirecting client requests).
    _election_timer / _heartbeat_timer:
        Handles returned by `scheduler.call_later` so timers can be cancelled
        or reset. Private because the tests do not rely on their exact type.
    """

    node_id: str
    peers: List[str]
    transport: Transport
    scheduler: Scheduler
    apply: Callable[[Any, int], None]
    state: RaftState = field(default=RaftState.FOLLOWER, init=False)
    current_term: int = field(default=0, init=False)
    voted_for: Optional[str] = field(default=None, init=False)
    log: List[LogEntry] = field(default_factory=list, init=False)
    commit_index: int = field(default=-1, init=False)
    last_applied: int = field(default=-1, init=False)
    next_index: Dict[str, int] = field(default_factory=dict, init=False)
    match_index: Dict[str, int] = field(default_factory=dict, init=False)
    leader_id: Optional[str] = field(default=None, init=False)
    _election_timer: Optional[SchedulerCancel] = field(default=None, init=False)
    _heartbeat_timer: Optional[SchedulerCancel] = field(default=None, init=False)

    def start(self) -> None:
        """
        Prepare the node for participation in Raft.

        Responsibilities (see Section 5.2 of the paper):

        #. Register the `on_message` handler with the transport so that
           inbound RPCs are delivered to this instance.
        #. Reset state as necessary (e.g., ensure leader-specific maps are
           cleared when starting as a follower).
        #. Schedule a randomized election timeout via `_reset_election_timer`
           so the node eventually transitions to a candidate if no leader is
           heard from.

        Suggested implementation sketch:

           self.transport.register(self.node_id, self.on_message)
           self.state = RaftState.FOLLOWER
           self.leader_id = None
           self._reset_election_timer()

        The concrete steps may differ, but capturing these responsibilities is
        essential to bootstrapping the node. The tests will fail until the logic
        meets the documented expectations.
        """

        # TODO:
        #  * register message handler
        #  * initialise follower state
        #  * reset election timer
        raise NotImplementedError

    def client_append(self, command: Any) -> None:
        """
        Append a client command to the replicated log.

        Only the leader should accept client writes. Followers should direct
        clients to the known leader by returning or forwarding the request.

        Expected workflow when this node is the leader:

        #. Append a `LogEntry` containing `(current_term, command)` to the local
           log.
        #. Update `next_index` / `match_index` bookkeeping if this is the first
           entry or if peers lag behind.
        #. Immediately send AppendEntries RPCs (heartbeats with payloads) to all
           followers so replication proceeds without waiting for the next
           periodic heartbeat. The payload should carry `prev_log_index`,
           `prev_log_term`, `entries`, and `leader_commit` as described in the
           paper.

        The helper method should raise an error or ignore commands when the node
        is not the leader; the exact behaviour can be tailored to the runtime
        but should be consistent.
        """

        # TODO:
        #  * check whether or not in the leader role
        #  * append entry
        #  * send AppendEntries to followers
        raise NotImplementedError

    def on_message(self, msg: Dict[str, Any]) -> None:
        """
        Dispatch inbound Raft RPCs to the appropriate handler.

        Raft exchanges two primary message types:

        `request_vote` / `request_vote_response`
            Used during elections. Followers decide whether to grant votes; the
            candidate tallies responses to determine leadership.

        `append_entries` / `append_entries_response`
            Leaders use AppendEntries for both heartbeats (empty `entries`)
            and log replication (one or more `LogEntry` records).

        Implementation outline:

        #. Inspect `msg["type"]` and branch accordingly.
        #. Handle term comparisons first: if the incoming `term` is greater
           than `current_term` the node must step down to follower and update
           `current_term` (Raft guarantees are rooted in monotonic term
           numbers).
        #. Delegate to helper methods such as `_handle_request_vote` or
           `_handle_append_entries` that you should implement.
        #. Ensure election timers are reset on valid leader activity and that
           responses get sent using `transport.send`.

        Following the structure in Figure 2 of the Raft paper makes the logic
        manageable. Thorough logging and comments often help with debugging.
        """

        # TODO:
        #  * branch on message type
        #  * apply term rules
        #  * call role-specific handlers
        raise NotImplementedError

    def stop(self) -> None:
        """
        Clean up timers and prepare the node to shut down or restart.

        Raft nodes may need to pause (e.g., when leaving a simulation or
        stepping down in tests). A minimal implementation should:

        #. Cancel outstanding election and heartbeat timers using the private
           helpers below.
        #. Optionally flush leader metadata so a later `start` call begins
           from the follower role with a fresh timeout.

        The function does not need to persist state; that responsibility lives
        with higher-level components if durability is desired.
        """

        # TODO:
        #  * cancel timers
        #  * reset transient leader state
        raise NotImplementedError

    # Helper hooks left for future implementation
    def _reset_election_timer(self) -> None:
        """
        Schedule the next election timeout.

        Requirements captured in Section 5.2 of the Raft paper:

        * Randomize the timeout between `T` and `2T` (or similar) to reduce the
          chance of split votes. Use the injected `scheduler` to register a
          callback that triggers the election routine.
        * Cancel any existing election timer before scheduling a new one to
          avoid duplicate callbacks firing.
        * The callback should transition the node to candidate (if still a
          follower) and initiate vote requests.

        The tests observe that a timer is scheduled, but do not mandate the
        exact randomness distribution. You can choose appropriate constants.
        """

        # TODO:
        #  * cancel old timer
        #  * compute randomized delay
        #  * schedule election callback
        raise NotImplementedError

    def _cancel_election_timer(self) -> None:
        """
        Stop the currently scheduled election timeout, if any.
        """

        # TODO:
        #  * call the stored cancel handle and clear it
        raise NotImplementedError

    def _reset_heartbeat_timer(self) -> None:
        """
        Schedule the next heartbeat for leaders.

        Heartbeats are simply AppendEntries RPCs with empty `entries` sent at
        a shorter, fixed interval (typically `T/2`). You should:

        #. Cancel the previous heartbeat timer.
        #. Register a new callback that broadcasts heartbeats to followers.
        #. Use `next_index` and `match_index` to decide which log entries to
           include when followers are behind.

        Followers generally should not schedule heartbeat timers; reset the
        election timeout instead when a legitimate leader contacts them.
        """

        # TODO:
        #  * cancel old timer
        #  * schedule periodic heartbeat callback for leaders
        raise NotImplementedError

    def _cancel_heartbeat_timer(self) -> None:
        """
        Cancel the periodic heartbeat scheduler, if active.
        """

        # TODO:
        #  * call the stored cancel handle and clear it
        raise NotImplementedError
