// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

// Reentrancy note: the AgriDAO contract below uses a hand-rolled
// `_LocalReentrancyGuard` in this prototype to avoid an OpenZeppelin
// dependency. The logic is functionally equivalent to the OZ
// ReentrancyGuard, but a production deployment should import the
// canonical OZ utility so auditors get the battle-tested version. See
// agribrain/contracts/README.md for the production checklist.
// (Comment intentionally kept outside any NatSpec block; the Solidity
// docstring parser treats @-prefixed tokens as NatSpec tags and chokes
// on the OZ import path.)

interface IAgentRegistry {
    struct Agent {
        bytes32 id;
        string role;
        string meta;
        bool active;
    }
    function agents(address) external view returns (bytes32, string memory, string memory, bool);
}

interface IPolicyStore {
    function setPolicy(bytes32 key, uint256 value) external;
}

/// @title AgriDAO - Consortium-style governance for supply chain policy updates
/// @notice Enables cooperative stakeholders to propose, vote on, and execute
///         policy parameter changes through a quorum-based governance process
///         with voting periods and timelock execution. Only registered active
///         agents (verified via AgentRegistry) can propose or vote. Execution
///         triggers PolicyStore updates that propagate to the decision engine.
/// @dev Minimal reentrancy guard pulled inline so the contracts dir
/// stays self-contained without an OpenZeppelin import. Equivalent to
/// OZ's ReentrancyGuard but with no external dependency.
abstract contract _LocalReentrancyGuard {
    uint256 private constant _NOT_ENTERED = 1;
    uint256 private constant _ENTERED = 2;
    uint256 private _status = _NOT_ENTERED;
    error ReentrantCall();

    modifier nonReentrant() {
        if (_status == _ENTERED) revert ReentrantCall();
        _status = _ENTERED;
        _;
        _status = _NOT_ENTERED;
    }
}

contract AgriDAO is _LocalReentrancyGuard {

    // -----------------------------------------------------------------
    // Enums
    // -----------------------------------------------------------------

    enum ProposalState {
        Pending,
        Active,
        Succeeded,
        Defeated,
        Queued,
        Executed,
        Expired
    }

    // -----------------------------------------------------------------
    // Structs
    // -----------------------------------------------------------------

    struct Proposal {
        uint256 id;
        address proposer;
        string description;
        bytes32 policyKey;
        uint256 policyValue;
        bool executed;
        uint256 yesVotes;
        uint256 noVotes;
        uint256 createdAt;
        uint256 totalVoters;
        ProposalState state;
        uint256 queuedAt;
        // Block.timestamp at which voting becomes possible. When
        // ``VOTING_DELAY`` is zero (default for the simulator) this
        // equals ``createdAt`` and the proposal is immediately Active,
        // matching the legacy lifecycle. When the delay is positive the
        // proposal is held in ``Pending`` until ``votingStartsAt`` is
        // reached, at which point ``activate(id)`` (or any state-aware
        // read via ``getEffectiveState``) promotes it to Active.
        uint256 votingStartsAt;
    }

    // -----------------------------------------------------------------
    // State variables
    // -----------------------------------------------------------------

    address public immutable owner;
    address public immutable policyStore;
    address public immutable agentRegistry;

    uint256 public QUORUM_THRESHOLD = 3;
    uint256 public VOTING_PERIOD = 86400;    // 24 hours
    uint256 public EXECUTION_DELAY = 3600;   // 1 hour
    /// @notice Optional delay between proposal creation and voting start.
    ///         0 (default) preserves the legacy "Active immediately on
    ///         propose" behaviour. Positive values hold the proposal in
    ///         the Pending state until ``votingStartsAt`` is reached.
    uint256 public VOTING_DELAY = 0;

    uint256 public nextId;
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public hasVoted;

    // -----------------------------------------------------------------
    // Events
    // -----------------------------------------------------------------

    event Proposed(uint256 indexed id, address proposer, string description, bytes32 policyKey, uint256 policyValue);
    event Voted(uint256 indexed id, address voter, bool support);
    event Finalized(uint256 indexed id, ProposalState newState);
    event Queued(uint256 indexed id);
    event Executed(uint256 indexed id);
    /// @notice Emitted when the owner changes a governance parameter.
    /// Off-chain indexers / monitors should subscribe to this event to
    /// detect quorum / voting-period / execution-delay / voting-delay
    /// changes; pre-2026-05 these setters were silent so a malicious
    /// owner could shrink quorum to 1 without leaving an on-chain
    /// trail in the event log. ``key`` is the keccak256 of the
    /// parameter name (e.g. ``keccak256("QUORUM_THRESHOLD")``).
    event ParamChanged(bytes32 indexed key, uint256 oldValue, uint256 newValue);

    // -----------------------------------------------------------------
    // Modifiers
    // -----------------------------------------------------------------

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier onlyRegisteredAgent() {
        // slither-disable-next-line unused-return
        (, , , bool active) = IAgentRegistry(agentRegistry).agents(msg.sender);
        require(active, "not a registered active agent");
        _;
    }

    // -----------------------------------------------------------------
    // Constructor
    // -----------------------------------------------------------------

    constructor(address _policyStore, address _agentRegistry) {
        owner = msg.sender;
        policyStore = _policyStore;
        agentRegistry = _agentRegistry;
    }

    // -----------------------------------------------------------------
    // Owner setters
    // -----------------------------------------------------------------

    function setQuorumThreshold(uint256 _quorum) external onlyOwner {
        uint256 old = QUORUM_THRESHOLD;
        QUORUM_THRESHOLD = _quorum;
        emit ParamChanged(keccak256("QUORUM_THRESHOLD"), old, _quorum);
    }

    function setVotingPeriod(uint256 _period) external onlyOwner {
        uint256 old = VOTING_PERIOD;
        VOTING_PERIOD = _period;
        emit ParamChanged(keccak256("VOTING_PERIOD"), old, _period);
    }

    function setExecutionDelay(uint256 _delay) external onlyOwner {
        uint256 old = EXECUTION_DELAY;
        EXECUTION_DELAY = _delay;
        emit ParamChanged(keccak256("EXECUTION_DELAY"), old, _delay);
    }

    function setVotingDelay(uint256 _delay) external onlyOwner {
        uint256 old = VOTING_DELAY;
        VOTING_DELAY = _delay;
        emit ParamChanged(keccak256("VOTING_DELAY"), old, _delay);
    }

    // -----------------------------------------------------------------
    // Governance functions
    // -----------------------------------------------------------------

    function propose(
        string calldata description,
        bytes32 policyKey,
        uint256 policyValue
    ) external onlyRegisteredAgent returns (uint256) {
        uint256 id = ++nextId;
        // Honor the optional VOTING_DELAY: when zero the proposal is
        // immediately Active (legacy behaviour preserved); otherwise it
        // is held in Pending until ``votingStartsAt`` is reached.
        ProposalState initialState = VOTING_DELAY == 0
            ? ProposalState.Active
            : ProposalState.Pending;
        proposals[id] = Proposal({
            id: id,
            proposer: msg.sender,
            description: description,
            policyKey: policyKey,
            policyValue: policyValue,
            executed: false,
            yesVotes: 0,
            noVotes: 0,
            createdAt: block.timestamp,
            totalVoters: 0,
            state: initialState,
            queuedAt: 0,
            votingStartsAt: block.timestamp + VOTING_DELAY
        });
        emit Proposed(id, msg.sender, description, policyKey, policyValue);
        return id;
    }

    /// @notice Promote a Pending proposal to Active once its voting
    ///         delay has elapsed. Anyone can call this to advance the
    ///         lifecycle; the on-chain check enforces the timestamp.
    /// @dev Reverts if the proposal is not Pending or if the delay has
    ///      not yet elapsed.
    function activate(uint256 id) external {
        Proposal storage p = proposals[id];
        require(p.state == ProposalState.Pending, "not pending");
        // slither-disable-next-line timestamp
        require(block.timestamp >= p.votingStartsAt, "voting not started");
        p.state = ProposalState.Active;
        emit Finalized(id, ProposalState.Active);
    }

    function vote(uint256 id, bool support) external onlyRegisteredAgent {
        Proposal storage p = proposals[id];
        require(p.state == ProposalState.Active, "not active");
        // slither-disable-next-line timestamp
        require(block.timestamp <= p.createdAt + VOTING_PERIOD, "voting ended");
        require(!hasVoted[id][msg.sender], "already voted");

        hasVoted[id][msg.sender] = true;
        p.totalVoters++;

        if (support) {
            p.yesVotes++;
        } else {
            p.noVotes++;
        }
        emit Voted(id, msg.sender, support);
    }

    function finalize(uint256 id) external {
        Proposal storage p = proposals[id];
        require(p.state == ProposalState.Active, "not active");
        // slither-disable-next-line timestamp
        require(block.timestamp > p.createdAt + VOTING_PERIOD, "voting not ended");

        if (p.totalVoters >= QUORUM_THRESHOLD && p.yesVotes > p.noVotes) {
            p.state = ProposalState.Succeeded;
        } else {
            p.state = ProposalState.Defeated;
        }
        emit Finalized(id, p.state);
    }

    function queue(uint256 id) external {
        Proposal storage p = proposals[id];
        require(p.state == ProposalState.Succeeded, "not succeeded");
        p.state = ProposalState.Queued;
        p.queuedAt = block.timestamp;
        emit Queued(id);
    }

    // slither-disable-next-line reentrancy-no-eth,reentrancy-events
    function execute(uint256 id) external nonReentrant {
        Proposal storage p = proposals[id];
        require(p.state == ProposalState.Queued, "not queued");
        // slither-disable-next-line timestamp
        require(block.timestamp >= p.queuedAt + EXECUTION_DELAY, "timelock active");

        // Checks-Effects-Interactions: write state BEFORE the external
        // call so a reentrant PolicyStore cannot see a half-finalised
        // proposal. nonReentrant adds a second line of defence.
        p.state = ProposalState.Executed;
        p.executed = true;
        IPolicyStore(policyStore).setPolicy(p.policyKey, p.policyValue);
        emit Executed(id);
    }

    // -----------------------------------------------------------------
    // View functions
    // -----------------------------------------------------------------

    function getProposal(uint256 id) external view returns (Proposal memory) {
        return proposals[id];
    }

    function getState(uint256 id) external view returns (ProposalState) {
        return proposals[id].state;
    }

    /// @notice State view that reflects the time-based Pending->Active
    ///         transition without requiring a write call to ``activate``.
    ///         Useful for off-chain indexers that want to render the
    ///         current effective state of a proposal.
    function getEffectiveState(uint256 id) external view returns (ProposalState) {
        Proposal storage p = proposals[id];
        if (p.state == ProposalState.Pending && block.timestamp >= p.votingStartsAt) {
            return ProposalState.Active;
        }
        return p.state;
    }

    function getHasVoted(uint256 id, address voter) external view returns (bool) {
        return hasVoted[id][voter];
    }
}
