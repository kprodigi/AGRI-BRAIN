// REVIEWER NOTE — 2026-04: this config now ships THREE network entries:
//
//   localhost   — single-node Hardhat for unit tests.
//   hardhat     — Hardhat's in-process VM (chainId 31337), used by the
//                 hardhat test runner.
//   permissioned — template for a permissioned EVM consortium
//                 (Hyperledger Besu QBFT, Quorum IBFT-2.0, or a Geth
//                 Clique chain). The URL and accounts are read from
//                 env vars so the secrets never land in version control.
//
// The contracts themselves (DecisionLogger.sol, ProvenanceRegistry.sol,
// SLCARewards.sol, AgriDAO.sol) are EVM-portable; the only chain-side
// distinction between localhost and a permissioned consortium is which
// addresses hold ADMIN_ROLE and the LOGGER_ROLE / ANCHORER_ROLE /
// REWARDER_ROLE / SLASHER_ROLE grants. ``deploy.js`` honours
// ``EXTRA_LOGGERS`` / ``EXTRA_ANCHORERS`` env vars at deploy time so
// the validator set on a permissioned chain receives its grants in the
// same transaction batch as deployment.

require('@nomicfoundation/hardhat-toolbox');

const PERMISSIONED_RPC = process.env.PERMISSIONED_RPC_URL || '';
const PERMISSIONED_CHAIN_ID = parseInt(process.env.PERMISSIONED_CHAIN_ID || '0', 10);
const PERMISSIONED_PRIVKEYS = (process.env.PERMISSIONED_PRIVKEYS || '')
  .split(',')
  .map((k) => k.trim())
  .filter((k) => k.length > 0);

// Build the networks dict so the ``permissioned`` key only appears
// when its env vars are set. Hardhat's config validator rejects an
// ``undefined`` network entry, so the consortium template is added
// dynamically rather than declared and left null.
const networks = {
  // Single-node Hardhat localhost. Used for the dev quickstart in
  // HOW_TO_RUN.md §7.
  localhost: { url: 'http://127.0.0.1:8545' },
  // Hardhat's in-process VM with a deterministic seed for tests.
  hardhat: { chainId: 31337 },
};

if (PERMISSIONED_RPC) {
  // Permissioned EVM consortium activation. Set the env vars below to
  // deploy against Hyperledger Besu QBFT, Quorum IBFT-2.0, or a Geth
  // Clique consortium:
  //   PERMISSIONED_RPC_URL=https://besu-validator-1.example.org:8545
  //   PERMISSIONED_CHAIN_ID=2025
  //   PERMISSIONED_PRIVKEYS=0xabc...,0xdef...   (comma-separated)
  //   EXTRA_LOGGERS=0xagent1Address,0xagent2Address
  //   EXTRA_ANCHORERS=0xexplainerServiceAddress
  // Then:
  //   npx hardhat run scripts/deploy.js --network permissioned
  // The deploy script grants LOGGER_ROLE / ANCHORER_ROLE to each
  // EXTRA_* address in the same transaction batch as deployment, so
  // the permissioned validator set is on-chain from block 1.
  networks.permissioned = {
    url: PERMISSIONED_RPC,
    accounts: PERMISSIONED_PRIVKEYS,
    gas: parseInt(process.env.PERMISSIONED_GAS || '8000000', 10),
    gasPrice: parseInt(process.env.PERMISSIONED_GASPRICE || '0', 10) || 'auto',
    timeout: parseInt(process.env.PERMISSIONED_TIMEOUT_MS || '60000', 10),
  };
  if (PERMISSIONED_CHAIN_ID) {
    networks.permissioned.chainId = PERMISSIONED_CHAIN_ID;
  }
}

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
  networks,
};
