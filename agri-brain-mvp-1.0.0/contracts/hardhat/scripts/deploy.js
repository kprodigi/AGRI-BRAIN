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
    return addr;
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
    addresses.AgentRegistry = await deploy('AgentRegistry');
    addresses.PolicyStore = await deploy('PolicyStore');
    addresses.DecisionLogger = await deploy('DecisionLogger');
    addresses.SLCARewards = await deploy('SLCARewards');
    addresses.AgriDAO = await deploy('AgriDAO');

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
