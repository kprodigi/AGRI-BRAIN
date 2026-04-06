const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("AgentRegistry", function () {
  let registry;
  let owner;
  let alice;

  beforeEach(async function () {
    [owner, alice] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("AgentRegistry");
    registry = await Factory.deploy();
    await registry.waitForDeployment();
  });

  it("registers an agent and marks it active", async function () {
    const id = ethers.id("agent:farm:alice");
    await expect(registry.connect(alice).register(id, "farm", "meta:v1"))
      .to.emit(registry, "Registered")
      .withArgs(alice.address, id, "farm", "meta:v1");

    const rec = await registry.agents(alice.address);
    expect(rec.id).to.equal(id);
    expect(rec.role).to.equal("farm");
    expect(rec.meta).to.equal("meta:v1");
    expect(rec.active).to.equal(true);
  });

  it("allows registered agent to toggle active status", async function () {
    const id = ethers.id("agent:processor:alice");
    await registry.connect(alice).register(id, "processor", "meta:v1");

    await expect(registry.connect(alice).setActive(false))
      .to.emit(registry, "Status")
      .withArgs(alice.address, false);
    expect((await registry.agents(alice.address)).active).to.equal(false);

    await expect(registry.connect(alice).setActive(true))
      .to.emit(registry, "Status")
      .withArgs(alice.address, true);
    expect((await registry.agents(alice.address)).active).to.equal(true);
  });

  it("reverts setActive for unregistered addresses", async function () {
    await expect(registry.connect(alice).setActive(false)).to.be.revertedWith("not registered");
  });

  it("overwrites fields on re-registration from same address", async function () {
    const id1 = ethers.id("agent:farm:alice:v1");
    const id2 = ethers.id("agent:farm:alice:v2");
    await registry.connect(alice).register(id1, "farm", "meta:v1");
    await registry.connect(alice).register(id2, "processor", "meta:v2");

    const rec = await registry.agents(alice.address);
    expect(rec.id).to.equal(id2);
    expect(rec.role).to.equal("processor");
    expect(rec.meta).to.equal("meta:v2");
    expect(rec.active).to.equal(true);
  });

  it("stores independent agent records for different signers", async function () {
    const idA = ethers.id("agent:a");
    const idB = ethers.id("agent:b");

    await registry.connect(alice).register(idA, "farm", "meta:a");
    await registry.connect(owner).register(idB, "recovery", "meta:b");

    const recA = await registry.agents(alice.address);
    const recB = await registry.agents(owner.address);
    expect(recA.id).to.equal(idA);
    expect(recA.role).to.equal("farm");
    expect(recB.id).to.equal(idB);
    expect(recB.role).to.equal("recovery");
  });
});
