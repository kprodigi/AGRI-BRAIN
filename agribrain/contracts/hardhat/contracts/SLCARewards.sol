// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title SLCARewards - Optional local integer-bookkeeping prototype
/// @notice Implements role-gated additions and deductions to an integer balance.
///         Inputs are caller supplied; this contract does not evaluate social
///         outcomes, waste, emissions, entitlement, or real-world performance.
/// @dev    Research code tested only with the local Hardhat suite. It is not a
///         token standard, payment system, production incentive layer, or audited
///         deployment component.
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
        // Local fixture convenience: the creator receives all three roles.
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
