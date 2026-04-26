// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title PolicyStore - On-chain governance parameter registry
/// @notice Stores policy thresholds that map to physical models in the backend:
///         - keccak256("max_temp_c"): Max cold-chain temperature (Celsius * 100).
///           Based on Arrhenius spoilage model: T > 8C triggers accelerated decay.
///         - keccak256("Ea_R"): Arrhenius activation energy / R (Kelvin * 100).
///           Ea_R = 8000 K corresponds to Ea ~ 66.5 kJ/mol for leafy greens.
///         - keccak256("carbon_per_km"): Transport emission factor (g CO2-eq/km).
///           Based on GHG Protocol vehicle emission factors (WRI/WBCSD, 2004).
///         - keccak256("waste_eta"): Waste penalty coefficient * 1000.
///           Controls reward trade-off between SLCA and waste reduction.
contract PolicyStore {
    address public immutable owner;
    address public authorizedDAO;

    mapping(bytes32 => uint256) public policy;

    /// @notice Per-key validation bounds. setPolicy reverts when value
    /// is outside [min, max]. Whitelist-only writes prevent the DAO
    /// from setting junk keys that would silently pollute on-chain
    /// storage. This addresses the reviewer concern that an unconstrained
    /// keyspace lets a passing proposal set arbitrary `bytes32` keys.
    struct KeyBound {
        uint256 minValue;
        uint256 maxValue;
        bool registered;
    }
    mapping(bytes32 => KeyBound) public keyBounds;
    bytes32[] public registeredKeys;

    event PolicyChanged(bytes32 indexed key, uint256 oldValue, uint256 newValue);
    event KeyRegistered(bytes32 indexed key, uint256 minValue, uint256 maxValue);

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    constructor() {
        owner = msg.sender;
        // Bootstrap canonical keys with reasonable bounds. Owner can
        // register additional keys at any time via registerKey.
        // max_temp_c is in Celsius*100; valid range 0..2000 (0..20 C).
        _register(keccak256("max_temp_c"), 0, 2000);
        // Ea_R in K*100; valid range 200000..1500000 (2000..15000 K).
        _register(keccak256("Ea_R"), 200000, 1500000);
        // carbon_per_km in g CO2-eq/km; valid range 50..500.
        _register(keccak256("carbon_per_km"), 50, 500);
        // waste_eta * 1000; valid range 0..2000 (0..2.0 dimensionless).
        _register(keccak256("waste_eta"), 0, 2000);
    }

    function _register(bytes32 key, uint256 minValue, uint256 maxValue) internal {
        require(minValue <= maxValue, "min>max");
        if (!keyBounds[key].registered) {
            registeredKeys.push(key);
        }
        keyBounds[key] = KeyBound({minValue: minValue, maxValue: maxValue, registered: true});
        emit KeyRegistered(key, minValue, maxValue);
    }

    function registerKey(bytes32 key, uint256 minValue, uint256 maxValue) external onlyOwner {
        _register(key, minValue, maxValue);
    }

    function setPolicy(bytes32 key, uint256 value) external {
        require(msg.sender == owner || msg.sender == authorizedDAO, "not authorized");
        KeyBound memory kb = keyBounds[key];
        require(kb.registered, "unregistered key");
        require(value >= kb.minValue && value <= kb.maxValue, "value out of range");
        uint256 oldv = policy[key];
        policy[key] = value;
        emit PolicyChanged(key, oldv, value);
    }

    function setAuthorizedDAO(address dao) external onlyOwner {
        authorizedDAO = dao;
    }

    function getPolicy(bytes32 key) external view returns (uint256) {
        return policy[key];
    }

    function registeredKeyCount() external view returns (uint256) {
        return registeredKeys.length;
    }
}
