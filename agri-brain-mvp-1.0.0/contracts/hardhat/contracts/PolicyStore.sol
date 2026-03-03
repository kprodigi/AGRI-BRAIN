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
    mapping(bytes32 => uint256) public policy;

    event PolicyChanged(bytes32 indexed key, uint256 oldValue, uint256 newValue);

    function setPolicy(bytes32 key, uint256 value) external {
        uint256 oldv = policy[key];
        policy[key] = value;
        emit PolicyChanged(key, oldv, value);
    }

    function getPolicy(bytes32 key) external view returns (uint256) {
        return policy[key];
    }
}
