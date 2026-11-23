const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("SLCARewards", function () {
  let rewards;
  let owner;
  let alice;
  let bob;
  let REWARDER_ROLE;
  let SLASHER_ROLE;
  let ADMIN_ROLE;

  beforeEach(async function () {
    [owner, alice, bob] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("SLCARewards");
    rewards = await Factory.deploy();
    await rewards.waitForDeployment();
    REWARDER_ROLE = await rewards.REWARDER_ROLE();
    SLASHER_ROLE = await rewards.SLASHER_ROLE();
    ADMIN_ROLE = await rewards.ADMIN_ROLE();
  });

  it("grants REWARDER_ROLE, SLASHER_ROLE, and ADMIN_ROLE to the deployer", async function () {
    expect(await rewards.hasRole(REWARDER_ROLE, owner.address)).to.equal(true);
    expect(await rewards.hasRole(SLASHER_ROLE, owner.address)).to.equal(true);
    expect(await rewards.hasRole(ADMIN_ROLE, owner.address)).to.equal(true);
    expect(await rewards.hasRole(REWARDER_ROLE, alice.address)).to.equal(false);
  });

  it("rewards balances when caller has REWARDER_ROLE", async function () {
    await expect(rewards.connect(owner).reward(alice.address, 50))
      .to.emit(rewards, "Rewarded")
      .withArgs(alice.address, 50, owner.address);
    expect(await rewards.balance(alice.address)).to.equal(50);
  });

  it("slashes balances without underflow when caller has SLASHER_ROLE", async function () {
    await rewards.connect(owner).reward(alice.address, 80);
    await expect(rewards.connect(owner).slash(alice.address, 30))
      .to.emit(rewards, "Slashed")
      .withArgs(alice.address, 30, owner.address);
    expect(await rewards.balance(alice.address)).to.equal(50);

    await rewards.connect(owner).slash(alice.address, 9999);
    expect(await rewards.balance(alice.address)).to.equal(0);
  });

  it("reverts reward/slash when caller lacks the required role", async function () {
    await expect(rewards.connect(bob).reward(alice.address, 10))
      .to.be.revertedWith("SLCARewards: missing role");
    await expect(rewards.connect(bob).slash(alice.address, 10))
      .to.be.revertedWith("SLCARewards: missing role");
  });

  it("admin can grant REWARDER_ROLE to a delegate", async function () {
    await expect(rewards.connect(owner).grantRole(REWARDER_ROLE, alice.address))
      .to.emit(rewards, "RoleGranted")
      .withArgs(REWARDER_ROLE, alice.address, owner.address);
    expect(await rewards.hasRole(REWARDER_ROLE, alice.address)).to.equal(true);
    await rewards.connect(alice).reward(bob.address, 12);
    expect(await rewards.balance(bob.address)).to.equal(12);
  });

  it("admin can revoke REWARDER_ROLE", async function () {
    await rewards.connect(owner).grantRole(REWARDER_ROLE, alice.address);
    await expect(rewards.connect(owner).revokeRole(REWARDER_ROLE, alice.address))
      .to.emit(rewards, "RoleRevoked")
      .withArgs(REWARDER_ROLE, alice.address, owner.address);
    await expect(rewards.connect(alice).reward(bob.address, 10))
      .to.be.revertedWith("SLCARewards: missing role");
  });

  it("non-admin cannot grant or revoke roles", async function () {
    await expect(rewards.connect(bob).grantRole(REWARDER_ROLE, alice.address))
      .to.be.revertedWith("SLCARewards: missing role");
    await expect(rewards.connect(bob).revokeRole(REWARDER_ROLE, owner.address))
      .to.be.revertedWith("SLCARewards: missing role");
  });

  it("accumulates rewards for repeated calls", async function () {
    await rewards.connect(owner).reward(alice.address, 50);
    await rewards.connect(owner).reward(alice.address, 30);
    expect(await rewards.balance(alice.address)).to.equal(80);
  });

  it("allows slashing zero-balance account without revert", async function () {
    await expect(rewards.connect(owner).slash(bob.address, 10))
      .to.emit(rewards, "Slashed")
      .withArgs(bob.address, 10, owner.address);
    expect(await rewards.balance(bob.address)).to.equal(0);
  });
});
