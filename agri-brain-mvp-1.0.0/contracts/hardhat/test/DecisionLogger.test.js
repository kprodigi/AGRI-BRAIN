const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("DecisionLogger", function () {
  let logger, owner;

  beforeEach(async function () {
    [owner] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("DecisionLogger");
    logger = await Factory.deploy();
    await logger.waitForDeployment();
  });

  it("should log a decision and emit DecisionLogged event", async function () {
    const tx = await logger.logDecision(
      1000,          // ts
      "farm:agent1", // agent
      "farm",        // role
      "cold_chain",  // action
      850,           // slca_milli (0.850 * 1000)
      1200,          // carbon_milli (1.200 kg * 1000)
      "test note"    // note
    );
    const receipt = await tx.wait();
    const event = receipt.logs[0];
    expect(event).to.not.be.undefined;
  });

  it("should store memo in mapping and allow read-back", async function () {
    const tx = await logger.logDecision(
      2000, "processor:agent2", "processor", "local_redistribute",
      920, 800, "redistribution test"
    );
    const receipt = await tx.wait();
    // Compute expected id
    const id = ethers.keccak256(
      ethers.AbiCoder.defaultAbiCoder().encode(
        ["uint256", "string", "string", "address"],
        [2000, "processor:agent2", "local_redistribute", owner.address]
      )
    );
    const memo = await logger.memos(id);
    expect(memo.ts).to.equal(2000n);
    expect(memo.agent).to.equal("processor:agent2");
    expect(memo.role).to.equal("processor");
    expect(memo.action).to.equal("local_redistribute");
    expect(memo.slca_milli).to.equal(920n);
    expect(memo.carbon_milli).to.equal(800n);
  });

  it("should produce different IDs for different decisions", async function () {
    const tx1 = await logger.logDecision(100, "a", "farm", "cold_chain", 500, 500, "");
    const tx2 = await logger.logDecision(200, "b", "processor", "recovery", 600, 600, "");
    const r1 = await tx1.wait();
    const r2 = await tx2.wait();
    // Both should succeed (different IDs)
    expect(r1.status).to.equal(1);
    expect(r2.status).to.equal(1);
  });
});
