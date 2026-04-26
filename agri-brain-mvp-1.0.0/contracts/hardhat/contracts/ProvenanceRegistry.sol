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
/// @dev    PROTOTYPE — single-key Ownable. Production deployments must
///         replace with role-based access control (OZ AccessControl)
///         and a permissioned EVM. See
///         `agri-brain-mvp-1.0.0/contracts/README.md`.
contract ProvenanceRegistry {
    address public immutable owner;

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

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    /// @notice Anchor a Merkle root hash for a decision explanation.
    /// @dev Append-only: re-anchoring an existing root reverts so the
    ///      "immutable audit trail" claim holds. Earlier revisions
    ///      silently overwrote the timestamp on duplicate roots, which
    ///      is incompatible with an audit-trail guarantee.
    /// @param merkleRoot The root hash of the evidence Merkle tree.
    /// @param decisionId The decision identifier (e.g., blockchain tx hash).
    function anchor(bytes32 merkleRoot, string calldata decisionId) external onlyOwner {
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
    /// @param merkleRoot The root hash to verify.
    /// @return exists True if the root has been anchored.
    /// @return timestamp The block timestamp when it was anchored.
    function verify(bytes32 merkleRoot) external view returns (bool exists, uint256 timestamp) {
        ProvenanceRecord storage r = records[merkleRoot];
        return (r.timestamp > 0, r.timestamp);
    }

    /// @notice Return the total number of anchored provenance records.
    function totalRecords() external view returns (uint256) {
        return rootHashes.length;
    }
}
