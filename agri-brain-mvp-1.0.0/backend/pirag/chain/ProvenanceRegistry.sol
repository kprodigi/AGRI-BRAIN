// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;
contract ProvenanceRegistry {
    struct Record {
        address submitter;
        uint256 timestamp;
        bytes32 root;
        string policyURI;
    }
    mapping(bytes32 => Record) public records;
    mapping(address => bool) public allowlist;
    uint256 public minIntervalSeconds = 5;
    mapping(address => uint256) public lastSubmit;
    event RootAnchored(address indexed who, bytes32 indexed root, string policyURI);
    modifier onlyAllowed() { require(allowlist[msg.sender], "NotAllowed"); _; }
    function setAllowlist(address who, bool allowed) external { allowlist[who] = allowed; }
    function setMinInterval(uint256 s) external { minIntervalSeconds = s; }
    function anchor(bytes32 root, string calldata policyURI) external onlyAllowed {
        require(block.timestamp >= lastSubmit[msg.sender] + minIntervalSeconds, "RateLimited");
        records[root] = Record(msg.sender, block.timestamp, root, policyURI);
        lastSubmit[msg.sender] = block.timestamp;
        emit RootAnchored(msg.sender, root, policyURI);
    }
}
