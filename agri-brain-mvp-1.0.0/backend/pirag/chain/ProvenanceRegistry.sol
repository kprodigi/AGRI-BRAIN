// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/// @title ProvenanceRegistry - Merkle root anchoring for evidence provenance
/// @notice Anchors Merkle roots of retrieval evidence on-chain, ensuring that
///         all supply chain decisions are traceable to their supporting data.
///         The policyURI links to off-chain policy metadata including SLCA
///         weights (UNEP/SETAC, 2009), Arrhenius parameters (Ea_R, k_ref),
///         and emission factors (GHG Protocol, WRI/WBCSD, 2004).
contract ProvenanceRegistry {
    struct Record {
        address submitter;
        uint256 timestamp;
        bytes32 root;
        string policyURI;
    }
    address public owner;
    mapping(bytes32 => Record) public records;
    mapping(address => bool) public allowlist;
    uint256 public minIntervalSeconds = 5;
    mapping(address => uint256) public lastSubmit;
    event RootAnchored(address indexed who, bytes32 indexed root, string policyURI);
    constructor() { owner = msg.sender; }
    modifier onlyOwner() { require(msg.sender == owner, "NotOwner"); _; }
    modifier onlyAllowed() { require(allowlist[msg.sender], "NotAllowed"); _; }
    function setAllowlist(address who, bool allowed) external onlyOwner { allowlist[who] = allowed; }
    function setMinInterval(uint256 s) external onlyOwner { minIntervalSeconds = s; }
    function anchor(bytes32 root, string calldata policyURI) external onlyAllowed {
        require(block.timestamp >= lastSubmit[msg.sender] + minIntervalSeconds, "RateLimited");
        records[root] = Record(msg.sender, block.timestamp, root, policyURI);
        lastSubmit[msg.sender] = block.timestamp;
        emit RootAnchored(msg.sender, root, policyURI);
    }
}
