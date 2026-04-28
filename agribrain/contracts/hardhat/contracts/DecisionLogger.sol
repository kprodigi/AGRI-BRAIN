// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title DecisionLogger - On-chain audit trail for supply chain routing decisions
/// @notice Records SLCA (UNEP/SETAC Social LCA, 2009) composite scores and
///         carbon footprint (GHG Protocol activity-based emissions, WRI/WBCSD,
///         2004) for each routing decision, enabling provenance verification.
///
/// @dev    Access control: role-based (ADMIN_ROLE / LOGGER_ROLE), mirroring
///         the SLCARewards pattern. The deployer is granted both roles at
///         construction so existing operational scripts keep working;
///         additional logger keys (e.g. one per supply-chain agent, or the
///         AgriDAO contract) can be added by an ADMIN_ROLE holder.
///         The legacy ``setAuthorized(addr, allowed)`` shim is retained
///         so the existing backend chain wrapper (``backend/src/chain/eth.py``)
///         and the deploy script keep their signatures; internally it
///         delegates to ``grantRole`` / ``revokeRole``.
///
/// @dev    Permissioned EVM. The contract itself is EVM-portable and is
///         intended to be deployed on a permissioned chain (Hyperledger
///         Besu QBFT / IBFT 2.0, Quorum, or a Geth Clique consortium).
///         The hardhat.config.cjs ships a ``permissioned`` network entry
///         that configures the deployer + a curated list of validator
///         addresses; see ``HOW_TO_RUN.md`` §7c. Localhost Hardhat is
///         still supported for unit testing.
contract DecisionLogger {
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant LOGGER_ROLE = keccak256("LOGGER_ROLE");

    address public immutable owner;
    mapping(bytes32 => mapping(address => bool)) private _roles;

    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    event DecisionLogged(
        bytes32 indexed id, // keccak of (counter, sender, ts, agent, role, action)
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
    /// without relying on archive-node event scans.
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

    /// @dev Counter for collision-free decision IDs.
    uint256 public decisionCounter;

    constructor() {
        owner = msg.sender;
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(LOGGER_ROLE, msg.sender);
    }

    // ---------------------------------------------------------------------
    // Role helpers (lightweight, mirrors the SLCARewards pattern)
    // ---------------------------------------------------------------------
    modifier onlyRole(bytes32 role) {
        require(_roles[role][msg.sender], "missing role");
        _;
    }

    function hasRole(bytes32 role, address account) external view returns (bool) {
        return _roles[role][account];
    }

    function grantRole(bytes32 role, address account) external onlyRole(ADMIN_ROLE) {
        _grantRole(role, account);
    }

    function revokeRole(bytes32 role, address account) external onlyRole(ADMIN_ROLE) {
        _revokeRole(role, account);
    }

    function _grantRole(bytes32 role, address account) internal {
        if (!_roles[role][account]) {
            _roles[role][account] = true;
            emit RoleGranted(role, account, msg.sender);
        }
    }

    function _revokeRole(bytes32 role, address account) internal {
        if (_roles[role][account]) {
            _roles[role][account] = false;
            emit RoleRevoked(role, account, msg.sender);
        }
    }

    // ---------------------------------------------------------------------
    // Backward-compatible shim. Older deploy scripts and the backend chain
    // wrapper call ``setAuthorized``. The shim keeps that surface intact
    // while delegating to the role layer so all access-control state is
    // visible on-chain through ``RoleGranted`` / ``RoleRevoked`` events.
    // ---------------------------------------------------------------------
    function setAuthorized(address who, bool allowed) external onlyRole(ADMIN_ROLE) {
        if (allowed) {
            _grantRole(LOGGER_ROLE, who);
        } else {
            _revokeRole(LOGGER_ROLE, who);
        }
    }

    function authorized(address who) external view returns (bool) {
        return _roles[LOGGER_ROLE][who];
    }

    // ---------------------------------------------------------------------
    // Logging entry points
    // ---------------------------------------------------------------------
    function logDecision(
        uint256 ts,
        string calldata agent,
        string calldata role,
        string calldata action,
        uint256 slca_milli,
        uint256 carbon_milli,
        string calldata note
    ) external onlyRole(LOGGER_ROLE) returns (bytes32 id) {
        decisionCounter += 1;
        id = keccak256(abi.encode(decisionCounter, msg.sender,
                                   ts, agent, role, action));
        // slither-disable-next-line incorrect-equality
        require(memos[id].ts == 0, "duplicate decision id");
        memos[id] = Memo(ts, agent, role, action, slca_milli, carbon_milli, note);
        emit DecisionLogged(id, ts, agent, role, action,
                            slca_milli, carbon_milli, note);
    }

    function logEpisode(
        bytes32 root,
        uint256 ts,
        string calldata mode,
        string calldata scenario,
        uint256 seed,
        uint256 n_records,
        string calldata note
    ) external onlyRole(LOGGER_ROLE) {
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
