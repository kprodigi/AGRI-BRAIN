// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title ProvenanceRegistry - On-chain anchoring for explanation provenance
/// @notice Stores Merkle root hashes linking the explanation engine's
///         feature-attribution outputs to their source evidence. Each
///         root is derived from SHA-256 hashes of cited evidence
///         passages (see pirag/provenance/merkle.py).
/// @dev    Called optionally by the piRAG explanation module after
///         generating a decision rationale. Provides an immutable,
///         tamper-evident audit trail for explanation outputs.
///
/// @dev    Access control: role-based (ADMIN_ROLE / ANCHORER_ROLE),
///         mirroring the SLCARewards pattern. The deployer is granted
///         both roles at construction; additional anchorer keys (e.g.
///         per-agent service accounts, or the AgriDAO governance
///         contract) can be added by an ADMIN_ROLE holder. The legacy
///         ``onlyOwner`` semantics are preserved for the deployer key.
///
/// @dev    Permissioned EVM. Deploys cleanly on Hyperledger Besu QBFT,
///         Quorum, or a Geth Clique consortium; ``hardhat.config.cjs``
///         ships a ``permissioned`` network entry. Localhost Hardhat
///         is still supported for unit testing.
contract ProvenanceRegistry {
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant ANCHORER_ROLE = keccak256("ANCHORER_ROLE");

    address public immutable owner;
    mapping(bytes32 => mapping(address => bool)) private _roles;

    struct ProvenanceRecord {
        bytes32 merkleRoot;
        uint256 timestamp;
        string decisionId;
        address submitter;
    }

    mapping(bytes32 => ProvenanceRecord) public records;
    bytes32[] public rootHashes;

    event ProvenanceAnchored(
        bytes32 indexed merkleRoot,
        string decisionId,
        address indexed submitter,
        uint256 timestamp
    );
    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    constructor() {
        owner = msg.sender;
        _grantRole(ADMIN_ROLE, msg.sender);
        _grantRole(ANCHORER_ROLE, msg.sender);
    }

    // ---------------------------------------------------------------------
    // Role helpers
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

    // Legacy ``onlyOwner`` modifier kept for any external script that
    // relied on the older API; it now delegates to the ADMIN_ROLE check
    // so role state and ownership stay consistent.
    modifier onlyOwner() {
        require(msg.sender == owner && _roles[ADMIN_ROLE][msg.sender], "not owner");
        _;
    }

    // ---------------------------------------------------------------------
    // Anchoring
    // ---------------------------------------------------------------------
    /// @notice Anchor a Merkle root hash for a decision explanation.
    /// @dev Append-only: re-anchoring an existing root reverts so the
    ///      "immutable audit trail" claim holds.
    /// @param merkleRoot The root hash of the evidence Merkle tree.
    /// @param decisionId The decision identifier (e.g., blockchain tx hash).
    function anchor(bytes32 merkleRoot, string calldata decisionId)
        external
        onlyRole(ANCHORER_ROLE)
    {
        // slither-disable-next-line incorrect-equality
        require(records[merkleRoot].timestamp == 0, "already anchored");
        records[merkleRoot] = ProvenanceRecord({
            merkleRoot: merkleRoot,
            timestamp: block.timestamp,
            decisionId: decisionId,
            submitter: msg.sender
        });
        rootHashes.push(merkleRoot);
        emit ProvenanceAnchored(merkleRoot, decisionId, msg.sender, block.timestamp);
    }

    /// @notice Verify that a Merkle root has been anchored.
    function verify(bytes32 merkleRoot) external view returns (bool exists, uint256 timestamp) {
        ProvenanceRecord storage r = records[merkleRoot];
        return (r.timestamp > 0, r.timestamp);
    }

    /// @notice Return the total number of anchored provenance records.
    function totalRecords() external view returns (uint256) {
        return rootHashes.length;
    }
}
