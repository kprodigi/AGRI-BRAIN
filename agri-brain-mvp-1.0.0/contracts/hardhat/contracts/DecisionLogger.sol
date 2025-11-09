// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract DecisionLogger {
    event DecisionLogged(
        bytes32 indexed id, // keccak of (ts, agent, action, msg.sender)
        uint256 ts,
        string agent,
        string role,
        string action,
        uint256 slca_milli, // slca * 1000
        uint256 carbon_milli, // kg * 1000
        string note
    );

    struct Memo {
        uint256 ts;
        string agent;
        string role;
        string action;
        uint256 slca_milli;
        uint256 carbon_milli;
        string note;
    }

    mapping(bytes32 => Memo) public memos;

    function logDecision(
        uint256 ts,
        string calldata agent,
        string calldata role,
        string calldata action,
        uint256 slca_milli,
        uint256 carbon_milli,
        string calldata note
    ) external returns (bytes32 id) {
        id = keccak256(abi.encode(ts, agent, action, msg.sender));
        memos[id] = Memo(
            ts,
            agent,
            role,
            action,
            slca_milli,
            carbon_milli,
            note
        );
        emit DecisionLogged(
            id,
            ts,
            agent,
            role,
            action,
            slca_milli,
            carbon_milli,
            note
        );
    }
}
