
from pirag.provenance.merkle import merkle_root
def test_merkle_root():
    xs = [
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "ca978112ca1bbdcafac231b39a23dc4da786eff8147c4e72b9807785afee48bb",
    ]
    r = merkle_root(xs); assert isinstance(r, str) and len(r) == 64
    print("Merkle root:", r)
