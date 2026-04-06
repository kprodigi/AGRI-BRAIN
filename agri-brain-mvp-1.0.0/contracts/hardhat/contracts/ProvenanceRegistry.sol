// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title ProvenanceRegistry - On-chain anchoring for explanation provenance
/// @notice Stores Merkle root hashes linking AI-generated explanations to their
///         source evidence. Each root is derived from SHA-256 hashes of cited
///         evidence passages (see pirag/provenance/merkle.py).
/// @dev Called optionally by the piRAG explanation module after generating
///      a decision rationale. Provides an immutable, tamper-evident audit trail
///      for explainability outputs.
contract ProvenanceRegistry {
    address public owner;

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
    /// @param merkleRoot The root hash of the evidence Merkle tree.
    /// @param decisionId The decision identifier (e.g., blockchain tx hash).
    function anchor(bytes32 merkleRoot, string calldata decisionId) external onlyOwner {
        bool isNewRoot = records[merkleRoot].timestamp == 0;
        records[merkleRoot] = ProvenanceRecord({
            merkleRoot: merkleRoot,
            timestamp: block.timestamp,
            decisionId: decisionId,
            submitter: msg.sender
        });
        if (isNewRoot) {
            rootHashes.push(merkleRoot);
        }
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
