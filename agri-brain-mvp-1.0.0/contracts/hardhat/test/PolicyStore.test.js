const { expect } = require("chai");
const { ethers } = require("hardhat");

describe("PolicyStore", function () {
  let store;
  let owner;
  let dao;
  let outsider;

  beforeEach(async function () {
    [owner, dao, outsider] = await ethers.getSigners();
    const Factory = await ethers.getContractFactory("PolicyStore");
    store = await Factory.deploy();
    await store.waitForDeployment();
  });

  it("allows owner to set and read policy values", async function () {
    const key = ethers.id("max_temp_c");
    await expect(store.connect(owner).setPolicy(key, 800))
      .to.emit(store, "PolicyChanged")
      .withArgs(key, 0, 800);
    expect(await store.getPolicy(key)).to.equal(800);
  });

  it("reverts policy updates from unauthorized callers", async function () {
    const key = ethers.id("waste_eta");
    await expect(store.connect(outsider).setPolicy(key, 500)).to.be.revertedWith("not authorized");
  });

  it("allows authorized DAO to set policy after owner approval", async function () {
    const key = ethers.id("carbon_per_km");
    await store.connect(owner).setAuthorizedDAO(dao.address);

    await expect(store.connect(dao).setPolicy(key, 120))
      .to.emit(store, "PolicyChanged")
      .withArgs(key, 0, 120);
    expect(await store.getPolicy(key)).to.equal(120);
  });

  it("reverts setAuthorizedDAO when caller is not owner", async function () {
    await expect(store.connect(outsider).setAuthorizedDAO(dao.address)).to.be.revertedWith("not owner");
  });

  it("emits previous value when overwriting a policy key", async function () {
    const key = ethers.id("Ea_R");
    await store.connect(owner).setPolicy(key, 800000);

    await expect(store.connect(owner).setPolicy(key, 810000))
      .to.emit(store, "PolicyChanged")
      .withArgs(key, 800000, 810000);
    expect(await store.getPolicy(key)).to.equal(810000);
  });
});
