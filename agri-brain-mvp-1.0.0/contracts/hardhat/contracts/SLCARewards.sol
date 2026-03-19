// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title SLCARewards - Token-based incentive layer for Social LCA performance
/// @notice Rewards agents whose routing decisions achieve high SLCA composite
///         scores (UNEP/SETAC Social LCA Guidelines, 2009). Slashes agents
///         whose decisions produce excessive waste or carbon emissions.
///         Reward amounts are proportional to the multi-objective reward:
///         R(t) = SLCA_composite - eta * waste_penalty (see policy.py).
/// @dev In production, restrict reward/slash to an authorized governance
///      contract. Current prototype uses onlyOwner for simplicity.
contract SLCARewards {
    address public owner;
    mapping(address => uint256) public balance;

    event Rewarded(address indexed to, uint256 amount);
    event Slashed(address indexed from, uint256 amount);

    constructor() {
        owner = msg.sender;
    }

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    function reward(address to, uint256 amount) external onlyOwner {
        balance[to] += amount;
        emit Rewarded(to, amount);
    }

    function slash(address from, uint256 amount) external onlyOwner {
        if (balance[from] >= amount) balance[from] -= amount;
        else balance[from] = 0;
        emit Slashed(from, amount);
    }
}
