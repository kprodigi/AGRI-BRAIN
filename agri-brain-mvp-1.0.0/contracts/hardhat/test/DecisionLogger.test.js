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

  it("should log a decision and emit DecisionLogged event with correct fields", async function () {
    const tx = await logger.logDecision(
      1000,          // ts
      "farm:agent1", // agent
      "farm",        // role
      "cold_chain",  // action
      850,           // slca_milli (0.850 * 1000)
      1200,          // carbon_milli (1.200 kg * 1000)
      "test note"    // note
    );
    // Decode and assert event fields (not just log existence)
    const receipt = await tx.wait();
    const iface = logger.interface;
    const parsed = iface.parseLog({ topics: receipt.logs[0].topics, data: receipt.logs[0].data });
    expect(parsed.name).to.equal("DecisionLogged");
    expect(parsed.args.ts).to.equal(1000n);
    expect(parsed.args.agent).to.equal("farm:agent1");
    expect(parsed.args.role).to.equal("farm");
    expect(parsed.args.action).to.equal("cold_chain");
    expect(parsed.args.slca_milli).to.equal(850n);
    expect(parsed.args.carbon_milli).to.equal(1200n);
    expect(parsed.args.note).to.equal("test note");
  });

  it("should revert logDecision from unauthorized caller", async function () {
    const [, outsider] = await ethers.getSigners();
    await expect(
      logger.connect(outsider).logDecision(1, "x", "y", "z", 0, 0, "")
    ).to.be.revertedWith("not authorized");
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
