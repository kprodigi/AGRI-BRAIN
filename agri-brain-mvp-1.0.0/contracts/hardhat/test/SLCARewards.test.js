const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("SLCARewards", function () {
  let rewards;
  let owner;
  let alice;
  let bob;

  beforeEach(async function () {
    [owner, alice, bob] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("SLCARewards");
    rewards = await Factory.deploy();
    await rewards.waitForDeployment();
  });

  it("rewards balances when called by owner", async function () {
    await expect(rewards.connect(owner).reward(alice.address, 50))
      .to.emit(rewards, "Rewarded")
      .withArgs(alice.address, 50);
    expect(await rewards.balance(alice.address)).to.equal(50);
  });

  it("slashes balances without underflow", async function () {
    await rewards.connect(owner).reward(alice.address, 80);
    await expect(rewards.connect(owner).slash(alice.address, 30))
      .to.emit(rewards, "Slashed")
      .withArgs(alice.address, 30);
    expect(await rewards.balance(alice.address)).to.equal(50);

    await rewards.connect(owner).slash(alice.address, 9999);
    expect(await rewards.balance(alice.address)).to.equal(0);
  });

  it("reverts reward/slash when caller is not owner", async function () {
    await expect(rewards.connect(bob).reward(alice.address, 10)).to.be.revertedWith("not owner");
    await expect(rewards.connect(bob).slash(alice.address, 10)).to.be.revertedWith("not owner");
  });

  it("accumulates rewards for repeated calls", async function () {
    await rewards.connect(owner).reward(alice.address, 50);
    await rewards.connect(owner).reward(alice.address, 30);
    expect(await rewards.balance(alice.address)).to.equal(80);
  });

  it("allows slashing zero-balance account without revert", async function () {
    await expect(rewards.connect(owner).slash(bob.address, 10))
      .to.emit(rewards, "Slashed")
      .withArgs(bob.address, 10);
    expect(await rewards.balance(bob.address)).to.equal(0);
  });
});
