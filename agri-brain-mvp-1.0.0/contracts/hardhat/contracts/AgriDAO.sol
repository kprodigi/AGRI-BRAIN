// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title AgriDAO - Decentralized governance for supply chain policy updates
/// @notice Enables cooperative stakeholders to propose and vote on policy
///         parameter changes (e.g., SLCA weights, Arrhenius model parameters,
///         carbon emission factors). Proposals are text-based and reference
///         the physical models they affect. Execution triggers PolicyStore
///         updates that propagate to the PINN-based decision engine.
contract AgriDAO {
    struct Proposal {
        uint256 id;
        address proposer;
        string text;
        bool executed;
        uint256 yes;
        uint256 no;
    }

    uint256 public nextId;
    mapping(uint256 => Proposal) public proposals;
    mapping(uint256 => mapping(address => bool)) public voted;

    event Proposed(uint256 id, address proposer, string text);
    event Voted(uint256 id, address voter, bool support);
    event Executed(uint256 id);

    function propose(string calldata text) external returns (uint256) {
        uint256 id = ++nextId;
        proposals[id] = Proposal({
            id: id,
            proposer: msg.sender,
            text: text,
            executed: false,
            yes: 0,
            no: 0
        });
        emit Proposed(id, msg.sender, text);
        return id;
    }

    function vote(uint256 id, bool support) external {
        require(!voted[id][msg.sender], "already voted");
        voted[id][msg.sender] = true;
        if (support) proposals[id].yes++;
        else proposals[id].no++;
        emit Voted(id, msg.sender, support);
    }

    function execute(uint256 id) external {
        proposals[id].executed = true;
        emit Executed(id);
    }
}
