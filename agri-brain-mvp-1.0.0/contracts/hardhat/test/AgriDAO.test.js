const { expect } = require("chai");
const { ethers } = require("hardhat");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("AgriDAO", function () {
  let store;
  let registry;
  let dao;
  let owner;
  let agent1;
  let agent2;
  let agent3;
  let outsider;

  async function registerAgent(signer, role) {
    const id = ethers.id(`agent:${role}:${signer.address}`);
    await registry.connect(owner).ownerRegister(signer.address, id, role, "meta");
    return id;
  }

  beforeEach(async function () {
    [owner, agent1, agent2, agent3, outsider] = await ethers.getSigners();

    const PolicyStore = await ethers.getContractFactory("PolicyStore");
    store = await PolicyStore.deploy();
    await store.waitForDeployment();

    const AgentRegistry = await ethers.getContractFactory("AgentRegistry");
    registry = await AgentRegistry.deploy();
    await registry.waitForDeployment();

    const AgriDAO = await ethers.getContractFactory("AgriDAO");
    dao = await AgriDAO.deploy(await store.getAddress(), await registry.getAddress());
    await dao.waitForDeployment();

    await store.connect(owner).setAuthorizedDAO(await dao.getAddress());

    await registerAgent(agent1, "farm");
    await registerAgent(agent2, "processor");
    await registerAgent(agent3, "cooperative");
  });

  it("allows registered agent to propose and records proposal fields", async function () {
    const key = ethers.id("max_temp_c");
    await expect(dao.connect(agent1).propose("raise threshold", key, 875))
      .to.emit(dao, "Proposed");

    const p = await dao.getProposal(1);
    expect(p.id).to.equal(1);
    expect(p.proposer).to.equal(agent1.address);
    expect(p.policyKey).to.equal(key);
    expect(p.policyValue).to.equal(875);
    expect(p.state).to.equal(1); // Active
  });

  it("reverts propose when caller is not a registered active agent", async function () {
    const key = ethers.id("max_temp_c");
    await expect(dao.connect(outsider).propose("bad", key, 1)).to.be.revertedWith(
      "not a registered active agent"
    );
  });

  it("runs full proposal lifecycle and executes policy update", async function () {
    const key = ethers.id("waste_eta");
    await dao.connect(agent1).propose("adjust waste penalty", key, 540);

    await expect(dao.connect(agent1).vote(1, true))
      .to.emit(dao, "Voted")
      .withArgs(1, agent1.address, true);
    await dao.connect(agent2).vote(1, true);
    await dao.connect(agent3).vote(1, false);

    await time.increase(24 * 60 * 60 + 1);
    await expect(dao.finalize(1)).to.emit(dao, "Finalized");
    expect(await dao.getState(1)).to.equal(2); // Succeeded

    await expect(dao.queue(1)).to.emit(dao, "Queued").withArgs(1);
    expect(await dao.getState(1)).to.equal(4); // Queued

    await time.increase(60 * 60 + 1);
    await expect(dao.execute(1)).to.emit(dao, "Executed").withArgs(1);

    expect(await dao.getState(1)).to.equal(5); // Executed
    expect(await store.getPolicy(key)).to.equal(540);
  });

  it("allows owner to update governance parameters and blocks non-owner", async function () {
    await dao.connect(owner).setQuorumThreshold(5);
    await dao.connect(owner).setVotingPeriod(7200);
    await dao.connect(owner).setExecutionDelay(120);

    expect(await dao.QUORUM_THRESHOLD()).to.equal(5);
    expect(await dao.VOTING_PERIOD()).to.equal(7200);
    expect(await dao.EXECUTION_DELAY()).to.equal(120);

    await expect(dao.connect(outsider).setQuorumThreshold(2)).to.be.revertedWith("not owner");
    await expect(dao.connect(outsider).setVotingPeriod(3600)).to.be.revertedWith("not owner");
    await expect(dao.connect(outsider).setExecutionDelay(60)).to.be.revertedWith("not owner");
  });

  it("reverts vote after voting period ends", async function () {
    await dao.connect(agent1).propose("late vote", ethers.id("late"), 1);
    const period = await dao.VOTING_PERIOD();
    const now = await time.latest();
    await time.increase(Number(period) + 1);
    expect(await time.latest()).to.be.greaterThan(now);

    await expect(dao.connect(agent2).vote(1, true)).to.be.revertedWith("voting ended");
  });

  it("reverts execute before timelock expires", async function () {
    await dao.connect(agent1).propose("timelock check", ethers.id("timelock"), 11);
    await dao.connect(agent1).vote(1, true);
    await dao.connect(agent2).vote(1, true);
    await dao.connect(agent3).vote(1, false);

    await time.increase(24 * 60 * 60 + 1);
    await dao.finalize(1);
    await dao.queue(1);

    await expect(dao.execute(1)).to.be.revertedWith("timelock active");
  });

  it("reverts queue on proposal that is not succeeded", async function () {
    await dao.connect(agent1).propose("queue active proposal", ethers.id("active"), 7);
    await expect(dao.queue(1)).to.be.revertedWith("not succeeded");
  });

  it("prevents duplicate votes and premature finalization", async function () {
    const key = ethers.id("carbon_per_km");
    await dao.connect(agent1).propose("change carbon factor", key, 130);
    await dao.connect(agent1).vote(1, true);

    await expect(dao.connect(agent1).vote(1, false)).to.be.revertedWith("already voted");
    await expect(dao.finalize(1)).to.be.revertedWith("voting not ended");
  });

  it("marks proposal defeated when quorum not met", async function () {
    await dao.connect(agent1).propose("too few voters", ethers.id("x"), 1);
    await dao.connect(agent1).vote(1, true);
    await time.increase(24 * 60 * 60 + 1);
    await dao.finalize(1);
    expect(await dao.getState(1)).to.equal(3); // Defeated
  });
});
