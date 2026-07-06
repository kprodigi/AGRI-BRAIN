const { expect } = require("chai");
const { ethers } = require("hardhat");
const { anyValue } = require("@nomicfoundation/hardhat-chai-matchers/withArgs");
const { time } = require("@nomicfoundation/hardhat-network-helpers");

describe("ProvenanceRegistry", function () {
  let registry;
  let owner;
  let outsider;

  beforeEach(async function () {
    [owner, outsider] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("ProvenanceRegistry");
    registry = await Factory.deploy();
    await registry.waitForDeployment();
  });

  it("anchors provenance and makes it verifiable", async function () {
    const root = ethers.id("proof:decision:1");
    const decisionId = "decision-001";

    await expect(registry.connect(owner).anchor(root, decisionId))
      .to.emit(registry, "ProvenanceAnchored")
      .withArgs(root, decisionId, owner.address, anyValue);

    const [exists, ts] = await registry.verify(root);
    expect(exists).to.equal(true);
    expect(ts).to.be.greaterThan(0);
    expect(await registry.totalRecords()).to.equal(1);

    const rec = await registry.records(root);
    expect(rec.merkleRoot).to.equal(root);
    expect(rec.decisionId).to.equal(decisionId);
    expect(rec.submitter).to.equal(owner.address);
  });

  it("returns false for unknown roots", async function () {
    const [exists, ts] = await registry.verify(ethers.id("proof:missing"));
    expect(exists).to.equal(false);
    expect(ts).to.equal(0);
  });

  it("reverts anchoring from non-owner", async function () {
    await expect(
      registry.connect(outsider).anchor(ethers.id("proof:decision:x"), "decision-x")
    ).to.be.revertedWith("missing role");
  });

  it("supports role-based delegation: ADMIN_ROLE can grant ANCHORER_ROLE", async function () {
    const ANCHORER = await registry.ANCHORER_ROLE();
    expect(await registry.hasRole(ANCHORER, outsider.address)).to.equal(false);

    await registry.connect(owner).grantRole(ANCHORER, outsider.address);
    expect(await registry.hasRole(ANCHORER, outsider.address)).to.equal(true);

    await expect(
      registry.connect(outsider).anchor(ethers.id("proof:decision:y"), "decision-y")
    ).to.emit(registry, "ProvenanceAnchored");

    await registry.connect(owner).revokeRole(ANCHORER, outsider.address);
    await expect(
      registry.connect(outsider).anchor(ethers.id("proof:decision:z"), "decision-z")
    ).to.be.revertedWith("missing role");
  });

  it("reverts on duplicate root to preserve append-only audit trail", async function () {
    const root = ethers.id("proof:dup");
    await registry.connect(owner).anchor(root, "decision-a");
    await expect(
      registry.connect(owner).anchor(root, "decision-b")
    ).to.be.revertedWith("already anchored");

    const rec = await registry.records(root);
    expect(rec.decisionId).to.equal("decision-a");
    expect(await registry.totalRecords()).to.equal(1);
  });

  it("anchors multiple unique roots and verifies each", async function () {
    const roots = [ethers.id("proof:r1"), ethers.id("proof:r2"), ethers.id("proof:r3")];
    await registry.connect(owner).anchor(roots[0], "decision-1");
    await registry.connect(owner).anchor(roots[1], "decision-2");
    await registry.connect(owner).anchor(roots[2], "decision-3");

    expect(await registry.totalRecords()).to.equal(3);
    const now = await time.latest();
    for (const root of roots) {
      const [exists, ts] = await registry.verify(root);
      expect(exists).to.equal(true);
      expect(ts).to.be.greaterThan(0);
      expect(ts).to.be.lte(now);
    }
  });

  it("returns anchored root via rootHashes index getter", async function () {
    const root = ethers.id("proof:indexed");
    await registry.connect(owner).anchor(root, "decision-indexed");
    expect(await registry.rootHashes(0)).to.equal(root);
  });
});
