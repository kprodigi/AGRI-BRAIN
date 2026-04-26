// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title SLCARewards - Token-based incentive layer for Social LCA performance
/// @notice Rewards agents whose routing decisions achieve high SLCA composite
///         scores (UNEP/SETAC Social LCA Guidelines, 2009). Slashes agents
///         whose decisions produce excessive waste or carbon emissions.
///         Reward amounts are proportional to the multi-objective reward:
///         R(t) = SLCA_composite - eta * waste_penalty (see policy.py).
///
/// @dev    PROTOTYPE — research code, not deployed to a permissioned EVM in
///         the published runs. See `agri-brain-mvp-1.0.0/contracts/README.md`
///         for the audit findings the 2026-04 hardening addresses:
///
///         - Replaces single-key Ownable with role-based access control:
///           ADMIN_ROLE manages role grants; REWARDER_ROLE can mint
///           rewards; SLASHER_ROLE can deduct from balances. The admin
///           cannot mint directly without first granting itself
///           REWARDER_ROLE — visible on-chain.
///         - The on-chain governance contract (AgriDAO) should be
///           granted REWARDER_ROLE in production so reward issuance is
///           gated by DAO proposals, not the deployer key.
///         - Production deployments must run on a permissioned EVM
///           (Hyperledger Besu QBFT / IBFT-2.0); the only deployment
///           target wired today is localhost Hardhat.
contract SLCARewards {
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    bytes32 public constant REWARDER_ROLE = keccak256("REWARDER_ROLE");
    bytes32 public constant SLASHER_ROLE = keccak256("SLASHER_ROLE");

    mapping(bytes32 => mapping(address => bool)) private _roles;
    mapping(address => uint256) public balance;

    event Rewarded(address indexed to, uint256 amount, address indexed by);
    event Slashed(address indexed from, uint256 amount, address indexed by);
    event RoleGranted(bytes32 indexed role, address indexed account, address indexed sender);
    event RoleRevoked(bytes32 indexed role, address indexed account, address indexed sender);

    constructor() {
        _grantRole(ADMIN_ROLE, msg.sender);
        // The deployer also gets REWARDER_ROLE / SLASHER_ROLE for
        // initial bring-up; production should revoke these and grant
        // them to the DAO contract.
        _grantRole(REWARDER_ROLE, msg.sender);
        _grantRole(SLASHER_ROLE, msg.sender);
    }

    modifier onlyRole(bytes32 role) {
        require(_roles[role][msg.sender], "SLCARewards: missing role");
        _;
    }

    function hasRole(bytes32 role, address account) external view returns (bool) {
        return _roles[role][account];
    }

    function grantRole(bytes32 role, address account) external onlyRole(ADMIN_ROLE) {
        _grantRole(role, account);
    }

    function revokeRole(bytes32 role, address account) external onlyRole(ADMIN_ROLE) {
        require(role != ADMIN_ROLE || account != msg.sender, "SLCARewards: cannot self-revoke admin");
        if (_roles[role][account]) {
            _roles[role][account] = false;
            emit RoleRevoked(role, account, msg.sender);
        }
    }

    function _grantRole(bytes32 role, address account) internal {
        if (!_roles[role][account]) {
            _roles[role][account] = true;
            emit RoleGranted(role, account, msg.sender);
        }
    }

    function reward(address to, uint256 amount) external onlyRole(REWARDER_ROLE) {
        balance[to] += amount;
        emit Rewarded(to, amount, msg.sender);
    }

    function slash(address from, uint256 amount) external onlyRole(SLASHER_ROLE) {
        if (balance[from] >= amount) balance[from] -= amount;
        else balance[from] = 0;
        emit Slashed(from, amount, msg.sender);
    }
}
