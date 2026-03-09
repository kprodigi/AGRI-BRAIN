// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title DecisionLogger - On-chain audit trail for supply chain routing decisions
/// @notice Records SLCA (UNEP/SETAC Social LCA, 2009) composite scores and
///         carbon footprint (GHG Protocol activity-based emissions, WRI/WBCSD, 2004)
///         for each routing decision, enabling provenance verification.
contract DecisionLogger {
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
