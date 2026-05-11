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

  // ---- Matrix-shaped policy parameters (Theta, Theta_context) ----

  // Paper-correct THETA_CONTEXT row entries from §3.8 (milli-scaled).
  // cold_chain row: [-0.80, -0.60, -0.15, -0.30, +0.25]
  // local_redist : [+0.50, +0.40, +0.20, +0.25, +0.10]
  // recovery     : [+0.30, +0.20, -0.05, +0.05, -0.35]
  const THETA_CONTEXT_MILLI_ROWMAJOR = [
    -800, -600, -150, -300, 250,
    500, 400, 200, 250, 100,
    300, 200, -50, 50, -350,
  ];

  it("registers THETA and THETA_CONTEXT matrix keys with the paper-spec shapes", async function () {
    const thetaCtxKey = ethers.id("THETA_CONTEXT");
    const thetaKey = ethers.id("THETA");

    const ctxBound = await store.matrixBounds(thetaCtxKey);
    expect(ctxBound.expectedRows).to.equal(3);
    expect(ctxBound.expectedCols).to.equal(5);
    expect(ctxBound.registered).to.equal(true);

    const baseBound = await store.matrixBounds(thetaKey);
    expect(baseBound.expectedRows).to.equal(3);
    expect(baseBound.expectedCols).to.equal(10);
    expect(baseBound.registered).to.equal(true);

    expect(await store.registeredMatrixKeyCount()).to.equal(2);
  });

  it("anchors THETA_CONTEXT matrix and round-trips the milli-scaled values", async function () {
    const key = ethers.id("THETA_CONTEXT");
    await expect(store.connect(owner).setPolicyMatrix(key, 3, 5, THETA_CONTEXT_MILLI_ROWMAJOR))
      .to.emit(store, "PolicyMatrixChanged")
      .withArgs(key, 1, 3, 5);

    const [rows, cols, vals, version] = await store.getPolicyMatrix(key);
    expect(rows).to.equal(3);
    expect(cols).to.equal(5);
    expect(version).to.equal(1);
    expect(vals.map((v) => Number(v))).to.deep.equal(THETA_CONTEXT_MILLI_ROWMAJOR);
  });

  it("monotone-bumps version on every setPolicyMatrix call", async function () {
    const key = ethers.id("THETA_CONTEXT");
    await store.connect(owner).setPolicyMatrix(key, 3, 5, THETA_CONTEXT_MILLI_ROWMAJOR);
    const tweaked = THETA_CONTEXT_MILLI_ROWMAJOR.slice();
    tweaked[0] = -750; // small drift
    await store.connect(owner).setPolicyMatrix(key, 3, 5, tweaked);
    const [, , , version] = await store.getPolicyMatrix(key);
    expect(version).to.equal(2);
  });

  it("reverts setPolicyMatrix on shape mismatch", async function () {
    const key = ethers.id("THETA_CONTEXT");
    await expect(
      store.connect(owner).setPolicyMatrix(key, 4, 5, THETA_CONTEXT_MILLI_ROWMAJOR)
    ).to.be.revertedWith("shape mismatch");
    await expect(
      store.connect(owner).setPolicyMatrix(key, 3, 4, THETA_CONTEXT_MILLI_ROWMAJOR)
    ).to.be.revertedWith("shape mismatch");
  });

  it("reverts setPolicyMatrix on length mismatch", async function () {
    const key = ethers.id("THETA_CONTEXT");
    const wrongLen = THETA_CONTEXT_MILLI_ROWMAJOR.slice(0, 14); // 14 != 3*5
    await expect(
      store.connect(owner).setPolicyMatrix(key, 3, 5, wrongLen)
    ).to.be.revertedWith("length mismatch");
  });

  it("reverts setPolicyMatrix on cell-magnitude breach", async function () {
    const key = ethers.id("THETA_CONTEXT");
    const explosive = THETA_CONTEXT_MILLI_ROWMAJOR.slice();
    explosive[7] = 9999; // >5000 (the registered maxAbsMilli for THETA_CONTEXT)
    await expect(
      store.connect(owner).setPolicyMatrix(key, 3, 5, explosive)
    ).to.be.revertedWith("matrix cell out of range");
  });

  it("reverts setPolicyMatrix on unregistered key", async function () {
    const key = ethers.id("UNREGISTERED_MATRIX");
    await expect(
      store.connect(owner).setPolicyMatrix(key, 3, 5, THETA_CONTEXT_MILLI_ROWMAJOR)
    ).to.be.revertedWith("unregistered matrix key");
  });

  it("reverts setPolicyMatrix from unauthorized caller", async function () {
    const key = ethers.id("THETA_CONTEXT");
    await expect(
      store.connect(outsider).setPolicyMatrix(key, 3, 5, THETA_CONTEXT_MILLI_ROWMAJOR)
    ).to.be.revertedWith("not authorized");
  });

  it("authorized DAO can also set policy matrices", async function () {
    const key = ethers.id("THETA_CONTEXT");
    await store.connect(owner).setAuthorizedDAO(dao.address);
    await expect(
      store.connect(dao).setPolicyMatrix(key, 3, 5, THETA_CONTEXT_MILLI_ROWMAJOR)
    ).to.emit(store, "PolicyMatrixChanged");
    const [rows, cols] = await store.getPolicyMatrix(key);
    expect(rows).to.equal(3);
    expect(cols).to.equal(5);
  });

  it("registerMatrixKey lets owner add additional matrix keys at runtime", async function () {
    const key = ethers.id("CUSTOM_MATRIX");
    await expect(store.connect(owner).registerMatrixKey(key, 2, 3, 1000))
      .to.emit(store, "MatrixKeyRegistered")
      .withArgs(key, 2, 3, 1000);
    await store.connect(owner).setPolicyMatrix(key, 2, 3, [10, 20, 30, 40, 50, 60]);
    const [, , vals] = await store.getPolicyMatrix(key);
    expect(vals.map((v) => Number(v))).to.deep.equal([10, 20, 30, 40, 50, 60]);
  });

  it("registerMatrixKey rejects matrices over MAX_MATRIX_CELLS=256", async function () {
    // 17 * 17 = 289 cells, above the 256 cap. The cap exists to
    // prevent gas-DoS via setPolicyMatrix's per-cell loop. Paper's
    // largest matrix is THETA at (3, 10) = 30 cells, so 256 leaves
    // generous headroom while keeping the loop bounded.
    const key = ethers.id("OVERSIZE_MATRIX");
    await expect(
      store.connect(owner).registerMatrixKey(key, 17, 17, 1000)
    ).to.be.revertedWith("matrix too large");
  });

  it("registerMatrixKey accepts matrices exactly at MAX_MATRIX_CELLS=256", async function () {
    // 16 * 16 = 256 cells, at the cap. Boundary inclusivity guard.
    const key = ethers.id("BOUNDARY_MATRIX");
    await expect(
      store.connect(owner).registerMatrixKey(key, 16, 16, 1000)
    )
      .to.emit(store, "MatrixKeyRegistered")
      .withArgs(key, 16, 16, 1000);
  });
});
