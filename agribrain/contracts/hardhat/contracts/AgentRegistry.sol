// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title AgentRegistry - Supply chain agent identity and capability registry
/// @notice Registers agents (farm, processor, cooperative, distributor, recovery) with their
///         roles and metadata. Active status controls participation in the multi-agent
///         decision system. Registration is gated by an existing operator-tier agent
///         (or the contract owner during bootstrap) to prevent Sybil capture of the
///         AgriDAO quorum, addressing the previous risk that any address
///         could self-register and immediately vote.
contract AgentRegistry {
    struct Agent {
        bytes32 id;
        string role;
        string meta;
        bool active;
    }

    address public owner;
    mapping(address => Agent) public agents;
    /// @notice Roles permitted to admit new agents. Set true during deploy.
    /// `cooperative` is the canonical operator tier; the owner can grant to others.
    mapping(string => bool) public adminRole;

    event Registered(address indexed agent, bytes32 id, string role, string meta);
    event Status(address indexed agent, bool active);
    event AdminRoleUpdated(string role, bool allowed);
    event Sponsored(address indexed sponsor, address indexed sponsee);

    error NotOwner();
    error NotAdmin();
    error AlreadyRegistered();
    error EmptyRole();

    error RoleNotLowercase();

    constructor() {
        owner = msg.sender;
        adminRole["cooperative"] = true;
    }

    modifier onlyOwner() {
        if (msg.sender != owner) revert NotOwner();
        _;
    }

    /// @dev Reject role strings that contain any uppercase ASCII letter.
    /// adminRole is keyed by string, so without normalisation
    /// ``adminRole["Cooperative"]`` and ``adminRole["cooperative"]`` are
    /// distinct keys -- a typo by the owner could create a silent
    /// privilege gap. This guard enforces all role keys to be lowercase
    /// at the registration site so the mapping has a single canonical
    /// key per role. Non-letter characters and digits are passed
    /// through unchanged so role strings can still carry domain
    /// suffixes (e.g. "cooperative-region-1").
    function _requireLowercaseRole(string memory role) internal pure {
        bytes memory b = bytes(role);
        for (uint256 i = 0; i < b.length; i++) {
            // ASCII A..Z is 0x41..0x5A.
            if (b[i] >= 0x41 && b[i] <= 0x5A) revert RoleNotLowercase();
        }
    }

    /// @notice Owner can extend or revoke admin authority on a role.
    function setAdminRole(string calldata role, bool allowed) external onlyOwner {
        _requireLowercaseRole(role);
        adminRole[role] = allowed;
        emit AdminRoleUpdated(role, allowed);
    }

    /// @notice Bootstrap registration by the contract owner (used to seed the
    /// initial cooperative agents during deployment). Once any agent with an
    /// admin role is active, the owner-only path is no longer required.
    function ownerRegister(
        address account,
        bytes32 id,
        string calldata role,
        string calldata meta
    ) external onlyOwner {
        _register(account, id, role, meta);
    }

    /// @notice Sponsored registration: an existing active agent with an admin
    /// role admits a new agent. This is the production path.
    function sponsorRegister(
        address account,
        bytes32 id,
        string calldata role,
        string calldata meta
    ) external {
        Agent storage sponsor = agents[msg.sender];
        if (!sponsor.active || !adminRole[sponsor.role]) revert NotAdmin();
        _register(account, id, role, meta);
        emit Sponsored(msg.sender, account);
    }

    function _register(
        address account,
        bytes32 id,
        string calldata role,
        string calldata meta
    ) internal {
        if (bytes(role).length == 0) revert EmptyRole();
        _requireLowercaseRole(role);
        if (agents[account].id != bytes32(0)) revert AlreadyRegistered();
        agents[account] = Agent({id: id, role: role, meta: meta, active: true});
        emit Registered(account, id, role, meta);
    }

    function setActive(bool on) external {
        require(agents[msg.sender].id != bytes32(0), "not registered");
        agents[msg.sender].active = on;
        emit Status(msg.sender, on);
    }

    /// @notice Owner can deactivate a misbehaving agent (slashing-lite).
    function ownerSetActive(address account, bool on) external onlyOwner {
        require(agents[account].id != bytes32(0), "not registered");
        agents[account].active = on;
        emit Status(account, on);
    }
}
