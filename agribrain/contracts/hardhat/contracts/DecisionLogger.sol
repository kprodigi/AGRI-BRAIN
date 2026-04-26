// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title DecisionLogger - On-chain audit trail for supply chain routing decisions
/// @notice Records SLCA (UNEP/SETAC Social LCA, 2009) composite scores and
///         carbon footprint (GHG Protocol activity-based emissions, WRI/WBCSD, 2004)
///         for each routing decision, enabling provenance verification.
/// @dev    PROTOTYPE — single-key Ownable. The deployer key has unilateral
///         control over `setAuthorized`, which lets it grant/revoke
///         decision-logging rights to arbitrary addresses. Production
///         deployments must replace this with role-based access control
///         (OZ AccessControl) and a permissioned EVM (Besu QBFT). See
///         `agribrain/contracts/README.md` for the production
///         checklist.
contract DecisionLogger {
    address public immutable owner;
    mapping(address => bool) public authorized;

    constructor() { owner = msg.sender; authorized[msg.sender] = true; }
    modifier onlyAuthorized() { require(authorized[msg.sender], "not authorized"); _; }
    function setAuthorized(address who, bool allowed) external {
        require(msg.sender == owner, "not owner");
        authorized[who] = allowed;
    }

    event DecisionLogged(
        bytes32 indexed id, // keccak of (ts, agent, action, msg.sender)
        uint256 ts,
        string agent,
        string role,
        string action,
        // SLCA composite score (UNEP/SETAC, 2009), scaled by 1000 for integer storage
        uint256 slca_milli,
        // Carbon footprint in kg CO2-eq (GHG Protocol), scaled by 1000
        uint256 carbon_milli,
        string note
    );

    /// @notice Per-episode anchor. The simulator collects every decision
    ///         within an episode into an off-chain ledger, computes a
    ///         binary SHA-256 Merkle root, and writes the root here in
    ///         a single transaction. Any individual decision can then
    ///         be verified with a Merkle inclusion proof against the
    ///         emitted root, giving full per-decision traceability at
    ///         O(1) on-chain cost per episode.
    event EpisodeLogged(
        bytes32 indexed root,
        uint256 ts,
        string mode,
        string scenario,
        uint256 seed,
        uint256 n_records,
        string note
    );

    struct Memo {
        uint256 ts;
        string agent;
        string role;
        string action;
        uint256 slca_milli;   // SLCA composite * 1000
        uint256 carbon_milli; // kg CO2-eq * 1000
        string note;
    }

    mapping(bytes32 => Memo) public memos;

    /// Episode-anchor storage so a verifier can look up an episode root
    /// without relying on archive-node event scans. The previous
    /// implementation only emitted EpisodeLogged events, which a
    /// reviewer correctly noted means light clients cannot prove the
    /// root was anchored after a chain reorg or after archive pruning.
    struct EpisodeAnchor {
        uint256 ts;
        string mode;
        string scenario;
        uint256 seed;
        uint256 n_records;
    }
    mapping(bytes32 => EpisodeAnchor) public episodeRoots;
    bytes32[] public episodeRootList;

    function episodeRootCount() external view returns (uint256) {
        return episodeRootList.length;
    }

    /// @dev Counter for collision-free decision IDs (the previous
    /// keccak(ts,agent,action,sender) collided when the simulator's
    /// timestep coarsened, silently dropping audit records).
    uint256 public decisionCounter;

    function logDecision(
        uint256 ts,
        string calldata agent,
        string calldata role,
        string calldata action,
        uint256 slca_milli,
        uint256 carbon_milli,
        string calldata note
    ) external onlyAuthorized returns (bytes32 id) {
        // Collision-free monotone ID derived from a counter so two
        // decisions with the same (ts, agent, action, sender) tuple but
        // different role/note do not silently revert.
        decisionCounter += 1;
        id = keccak256(abi.encode(decisionCounter, msg.sender,
                                   ts, agent, role, action));
        // slither-disable-next-line incorrect-equality
        require(memos[id].ts == 0, "duplicate decision id");
        memos[id] = Memo(ts, agent, role, action, slca_milli, carbon_milli, note);
        emit DecisionLogged(id, ts, agent, role, action,
                            slca_milli, carbon_milli, note);
    }

    /// @notice Anchor a Merkle root over all decisions in one episode.
    /// @dev Persists the root in storage AND emits ``EpisodeLogged`` so
    ///      light clients can verify anchoring without archive-node
    ///      access.
    function logEpisode(
        bytes32 root,
        uint256 ts,
        string calldata mode,
        string calldata scenario,
        uint256 seed,
        uint256 n_records,
        string calldata note
    ) external onlyAuthorized {
        // slither-disable-next-line incorrect-equality
        require(episodeRoots[root].ts == 0, "duplicate episode root");
        episodeRoots[root] = EpisodeAnchor({
            ts: ts,
            mode: mode,
            scenario: scenario,
            seed: seed,
            n_records: n_records
        });
        episodeRootList.push(root);
        emit EpisodeLogged(root, ts, mode, scenario, seed, n_records, note);
    }
}
