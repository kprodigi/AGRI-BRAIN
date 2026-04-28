// contracts/hardhat/scripts/deploy.js
const fs = require('fs');
const path = require('path');
const hre = require('hardhat');

async function deploy(name, ...args) {
    const F = await hre.ethers.getContractFactory(name);
    const c = await F.deploy(...args);
    await c.waitForDeployment();
    const addr = await c.getAddress();
    console.log(`${name}: ${addr}`);
    return { contract: c, address: addr };
}

function writeJSON(p, obj) {
    fs.mkdirSync(path.dirname(p), { recursive: true });
    fs.writeFileSync(p, JSON.stringify(obj, null, 2));
    console.log('Saved:', p);
}

async function main() {
    const [deployer] = await hre.ethers.getSigners();
    console.log('Network :', hre.network.name);
    console.log('Deployer:', deployer.address);

    const addresses = {};

    // 1. AgentRegistry first (no dependencies)
    const registry = await deploy('AgentRegistry');
    addresses.AgentRegistry = registry.address;

    // 2. PolicyStore second (no dependencies)
    const policyStore = await deploy('PolicyStore');
    addresses.PolicyStore = policyStore.address;

    // 3. AgriDAO third (receives PolicyStore + AgentRegistry addresses)
    const dao = await deploy('AgriDAO', policyStore.address, registry.address);
    addresses.AgriDAO = dao.address;

    // 4. Authorize AgriDAO to update policies via PolicyStore
    await policyStore.contract.setAuthorizedDAO(dao.address);
    console.log('PolicyStore: authorized AgriDAO at', dao.address);

    // 5. DecisionLogger and SLCARewards as before
    const logger = await deploy('DecisionLogger');
    addresses.DecisionLogger = logger.address;

    const rewards = await deploy('SLCARewards');
    addresses.SLCARewards = rewards.address;

    // 6. ProvenanceRegistry for Merkle root anchoring
    const provenance = await deploy('ProvenanceRegistry');
    addresses.ProvenanceRegistry = provenance.address;

    // 7. Role grants for the permissioned EVM workflow.
    // EXTRA_LOGGERS  -> grants LOGGER_ROLE on DecisionLogger so each
    //                   listed agent / service account can write
    //                   decision records.
    // EXTRA_ANCHORERS -> grants ANCHORER_ROLE on ProvenanceRegistry so
    //                   each listed explainer key can anchor Merkle
    //                   roots.
    const splitCsv = (s) => (s || '').split(',').map((x) => x.trim()).filter(Boolean);
    const extraLoggers = splitCsv(process.env.EXTRA_LOGGERS);
    const extraAnchorers = splitCsv(process.env.EXTRA_ANCHORERS);

    if (extraLoggers.length > 0) {
        const LOGGER_ROLE = await logger.contract.LOGGER_ROLE();
        for (const addr of extraLoggers) {
            await logger.contract.grantRole(LOGGER_ROLE, addr);
            console.log('DecisionLogger.LOGGER_ROLE granted to', addr);
        }
    }
    if (extraAnchorers.length > 0) {
        const ANCHORER_ROLE = await provenance.contract.ANCHORER_ROLE();
        for (const addr of extraAnchorers) {
            await provenance.contract.grantRole(ANCHORER_ROLE, addr);
            console.log('ProvenanceRegistry.ANCHORER_ROLE granted to', addr);
        }
    }

    // 8. Seed the canonical THETA / THETA_CONTEXT matrices on PolicyStore
    //    so a verifier can read them straight off-chain without waiting
    //    for the first DAO proposal. Values are milli-scaled (multiply
    //    floats by 1000) and the row-major layout matches the paper:
    //    THETA is (3, 6) over (cold_chain, local_redistribute, recovery)
    //    x (1-rho, inv, y_hat, thermal, rho, rho*inv); THETA_CONTEXT is
    //    (3, 5) over the same actions x (psi_0..psi_4). Re-seeding via
    //    setPolicyMatrix is idempotent and bumps the on-chain version
    //    counter so a verifier can detect overwrites.
    const THETA_MILLI = [
        // cold_chain row
        500, -300, 400, -500, -2000, -1000,
        // local_redistribute row
        0, 500, -200, 500, 2000, 1500,
        // recovery row
        -500, -300, -200, 300, 1500, -300,
    ];
    const THETA_CONTEXT_MILLI = [
        // cold_chain row
        -800, -600, -150, -300, 250,
        // local_redistribute row
        500, 400, 200, 250, 100,
        // recovery row
        300, 200, -50, 50, -350,
    ];
    const thetaKey = hre.ethers.id('THETA');
    const thetaCtxKey = hre.ethers.id('THETA_CONTEXT');
    await policyStore.contract.setPolicyMatrix(thetaKey, 3, 6, THETA_MILLI);
    console.log('PolicyStore.THETA seeded (3x6, milli-scaled)');
    await policyStore.contract.setPolicyMatrix(thetaCtxKey, 3, 5, THETA_CONTEXT_MILLI);
    console.log('PolicyStore.THETA_CONTEXT seeded (3x5, milli-scaled)');

    // Write inside hardhat dir
    const here = path.resolve(__dirname, '..', `deployed-addresses.${hre.network.name}.json`);
    writeJSON(here, addresses);

    // Also write next to the backend so it can auto-import
    const toBackend = path.resolve(__dirname, '..', '..', '..', 'backend', 'runtime', 'chain', `deployed-addresses.${hre.network.name}.json`);
    writeJSON(toBackend, addresses);

    // Stable "latest" symlink-ish file name for easy watching
    const latest = path.resolve(path.dirname(toBackend), 'deployed-addresses.latest.json');
    writeJSON(latest, { network: hre.network.name, addresses });

    console.log('\nAddresses JSON (paste not needed if auto-sync is on):');
    console.log(JSON.stringify(addresses, null, 2));
}

main().catch(e => { console.error(e); process.exit(1); });
