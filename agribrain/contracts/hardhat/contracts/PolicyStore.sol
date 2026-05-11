// SPDX-License-Identifier: MIT
pragma solidity ^0.8.28;

/// @title PolicyStore - On-chain governance parameter registry
/// @notice Stores both scalar policy thresholds and matrix-shaped policy
///         parameters that map to physical models in the backend.
///
///         Scalar keys (uint256, mapped via ``policy``):
///         - keccak256("max_temp_c"): Max cold-chain temperature (Celsius * 100).
///           Based on Arrhenius spoilage model: T > 8C triggers accelerated decay.
///         - keccak256("Ea_R"): Arrhenius activation energy / R (Kelvin * 100).
///           Ea_R = 8000 K corresponds to Ea ~ 66.5 kJ/mol for leafy greens.
///         - keccak256("carbon_per_km"): Transport emission factor (g CO2-eq/km).
///           Based on GHG Protocol vehicle emission factors (WRI/WBCSD, 2004).
///         - keccak256("waste_eta"): Waste penalty coefficient * 1000.
///           Controls reward trade-off between SLCA and waste reduction.
///
///         Matrix keys (int256[], mapped via ``policyMatrix``):
///         - keccak256("THETA"):           base policy weights, 3x10, milli-scaled.
///         - keccak256("THETA_CONTEXT"):   context weight matrix, 3x5, milli-scaled.
///         Each cell is stored as int256 (signed) scaled by 1000, so the
///         range +/-2.000 the paper uses for Theta entries is representable
///         exactly without floating-point round-trip. Reads return the
///         flat int256[] plus the (rows, cols) shape so off-chain readers
///         can reshape without an out-of-band schema.
contract PolicyStore {
    address public immutable owner;
    address public authorizedDAO;

    mapping(bytes32 => uint256) public policy;

    /// @notice Per-key validation bounds. setPolicy reverts when value
    /// is outside [min, max]. Whitelist-only writes prevent the DAO
    /// from setting junk keys that would silently pollute on-chain
    /// storage.
    struct KeyBound {
        uint256 minValue;
        uint256 maxValue;
        bool registered;
    }
    mapping(bytes32 => KeyBound) public keyBounds;
    bytes32[] public registeredKeys;

    /// @notice Matrix-shaped policy parameter (e.g. THETA, THETA_CONTEXT).
    ///         Stored as a flat int256 array plus an explicit (rows, cols)
    ///         shape; values are milli-scaled (multiply by 1000 off-chain
    ///         when writing, divide by 1000 when reading) so signed
    ///         decimals down to 1e-3 round-trip exactly. ``version`` is a
    ///         monotone counter incremented on every successful set so a
    ///         verifier can detect overwrites without scanning events.
    struct MatrixEntry {
        uint256 rows;
        uint256 cols;
        int256[] values;     // length == rows * cols, row-major
        uint256 version;
        bool registered;
    }
    mapping(bytes32 => MatrixEntry) private _policyMatrix;
    bytes32[] public registeredMatrixKeys;

    /// @notice Per-matrix-key absolute-value bound on each cell. setPolicyMatrix
    ///         reverts if any |cell| > maxAbsMilli. Matches the scalar
    ///         keyspace's whitelist + bound discipline so a misbehaving
    ///         DAO proposal cannot anchor pathological weights.
    struct MatrixBound {
        uint256 expectedRows;
        uint256 expectedCols;
        uint256 maxAbsMilli;
        bool registered;
    }
    mapping(bytes32 => MatrixBound) public matrixBounds;

    event PolicyChanged(bytes32 indexed key, uint256 oldValue, uint256 newValue);
    event KeyRegistered(bytes32 indexed key, uint256 minValue, uint256 maxValue);
    event PolicyMatrixChanged(
        bytes32 indexed key,
        uint256 indexed version,
        uint256 rows,
        uint256 cols
    );
    event MatrixKeyRegistered(
        bytes32 indexed key,
        uint256 expectedRows,
        uint256 expectedCols,
        uint256 maxAbsMilli
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "not owner");
        _;
    }

    modifier onlyAuthorized() {
        require(msg.sender == owner || msg.sender == authorizedDAO, "not authorized");
        _;
    }

    constructor() {
        owner = msg.sender;
        // Scalar keys with reasonable bounds.
        _register(keccak256("max_temp_c"), 0, 2000);
        _register(keccak256("Ea_R"), 200000, 1500000);
        _register(keccak256("carbon_per_km"), 50, 500);
        _register(keccak256("waste_eta"), 0, 2000);

        // Matrix keys for the canonical policy weight matrices. Both use
        // a maxAbsMilli of 5000 so cells in [-5.000, +5.000] are
        // accepted; the paper's largest single-cell magnitudes (Theta
        // cold-chain spoilage = -2.00, Theta_context cold-chain
        // compliance = -0.80) sit comfortably inside that envelope and
        // any proposal that drifts past +/-5.0 is gated.
        _registerMatrix(keccak256("THETA"), 3, 10, 5000);
        _registerMatrix(keccak256("THETA_CONTEXT"), 3, 5, 5000);
    }

    function _register(bytes32 key, uint256 minValue, uint256 maxValue) internal {
        require(minValue <= maxValue, "min>max");
        if (!keyBounds[key].registered) {
            registeredKeys.push(key);
        }
        keyBounds[key] = KeyBound({minValue: minValue, maxValue: maxValue, registered: true});
        emit KeyRegistered(key, minValue, maxValue);
    }

    /// @dev Hard upper bound on matrix cell count to prevent gas-DoS via
    /// pathologically-large matrices. The paper's largest matrix is THETA
    /// at (3, 10) = 30 cells; THETA_CONTEXT is (3, 5) = 15. The 256-cell
    /// cap leaves >10x headroom for future expansion while making
    /// setPolicyMatrix's per-cell loop bounded by a small constant. A
    /// matrix above this cap is rejected at registration time so a
    /// caller cannot register a key that would later DoS setPolicyMatrix.
    uint256 internal constant MAX_MATRIX_CELLS = 256;

    function _registerMatrix(
        bytes32 key,
        uint256 expectedRows,
        uint256 expectedCols,
        uint256 maxAbsMilli
    ) internal {
        require(expectedRows > 0 && expectedCols > 0, "shape>0");
        require(expectedRows * expectedCols <= MAX_MATRIX_CELLS, "matrix too large");
        if (!matrixBounds[key].registered) {
            registeredMatrixKeys.push(key);
        }
        matrixBounds[key] = MatrixBound({
            expectedRows: expectedRows,
            expectedCols: expectedCols,
            maxAbsMilli: maxAbsMilli,
            registered: true
        });
        emit MatrixKeyRegistered(key, expectedRows, expectedCols, maxAbsMilli);
    }

    function registerKey(bytes32 key, uint256 minValue, uint256 maxValue) external onlyOwner {
        _register(key, minValue, maxValue);
    }

    function registerMatrixKey(
        bytes32 key,
        uint256 expectedRows,
        uint256 expectedCols,
        uint256 maxAbsMilli
    ) external onlyOwner {
        _registerMatrix(key, expectedRows, expectedCols, maxAbsMilli);
    }

    function setPolicy(bytes32 key, uint256 value) external onlyAuthorized {
        KeyBound memory kb = keyBounds[key];
        require(kb.registered, "unregistered key");
        require(value >= kb.minValue && value <= kb.maxValue, "value out of range");
        uint256 oldv = policy[key];
        policy[key] = value;
        emit PolicyChanged(key, oldv, value);
    }

    /// @notice Anchor a matrix-shaped policy parameter (Theta, Theta_context, ...).
    /// @param key             keccak256 of the matrix's name (e.g.
    ///                        ``keccak256("THETA_CONTEXT")``).
    /// @param rows            number of rows; must match the registered shape.
    /// @param cols            number of columns; must match the registered shape.
    /// @param valuesMilli     row-major flat array of length rows*cols,
    ///                        each cell scaled by 1000 (so the paper's
    ///                        +0.50 becomes 500, -0.80 becomes -800).
    function setPolicyMatrix(
        bytes32 key,
        uint256 rows,
        uint256 cols,
        int256[] calldata valuesMilli
    ) external onlyAuthorized {
        MatrixBound memory mb = matrixBounds[key];
        require(mb.registered, "unregistered matrix key");
        require(rows == mb.expectedRows && cols == mb.expectedCols, "shape mismatch");
        require(valuesMilli.length == rows * cols, "length mismatch");
        // Cell-magnitude bound: |cell| must be <= maxAbsMilli.
        int256 maxAbs = int256(mb.maxAbsMilli);
        for (uint256 i = 0; i < valuesMilli.length; i++) {
            int256 v = valuesMilli[i];
            int256 abs = v >= 0 ? v : -v;
            require(abs <= maxAbs, "matrix cell out of range");
        }

        MatrixEntry storage entry = _policyMatrix[key];
        entry.rows = rows;
        entry.cols = cols;
        entry.values = valuesMilli;  // copies calldata into storage
        entry.version += 1;
        entry.registered = true;
        emit PolicyMatrixChanged(key, entry.version, rows, cols);
    }

    function setAuthorizedDAO(address dao) external onlyOwner {
        authorizedDAO = dao;
    }

    function getPolicy(bytes32 key) external view returns (uint256) {
        return policy[key];
    }

    /// @notice Return the matrix-shaped policy parameter at ``key``.
    /// @return rows           number of rows
    /// @return cols           number of columns
    /// @return valuesMilli    row-major flat array of length rows*cols,
    ///                        each cell milli-scaled (divide by 1000
    ///                        off-chain to recover the float value)
    /// @return version        monotone version counter, incremented on
    ///                        every successful set
    function getPolicyMatrix(bytes32 key)
        external
        view
        returns (uint256 rows, uint256 cols, int256[] memory valuesMilli, uint256 version)
    {
        MatrixEntry storage entry = _policyMatrix[key];
        return (entry.rows, entry.cols, entry.values, entry.version);
    }

    function registeredKeyCount() external view returns (uint256) {
        return registeredKeys.length;
    }

    function registeredMatrixKeyCount() external view returns (uint256) {
        return registeredMatrixKeys.length;
    }
}
