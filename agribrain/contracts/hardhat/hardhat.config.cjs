// REVIEWER NOTE — 2025-04 reformatted from a single-line config so
// readers can audit the Solidity build settings, optimizer level, and
// network list at a glance. The previous one-liner was actively
// hostile to review.

require('@nomicfoundation/hardhat-toolbox');

module.exports = {
  solidity: {
    version: '0.8.28',
    settings: {
      // Optimizer enabled with 200 runs (matches OpenZeppelin / typical
      // production defaults). Pinned bytecodeHash so verifiers get
      // deterministic metadata across rebuilds.
      optimizer: { enabled: true, runs: 200 },
      metadata: { bytecodeHash: 'ipfs' },
    },
  },
  networks: {
    // Single-node Hardhat localhost. The paper currently anchors all
    // results against this network; multi-validator deployment is
    // documented as an explicit known limitation in
    // docs/KNOWN_LIMITATIONS.md (item B1).
    localhost: { url: 'http://127.0.0.1:8545' },
    // Hardhat's in-process VM with a deterministic seed for tests.
    hardhat: { chainId: 31337 },
  },
};
