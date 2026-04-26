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
