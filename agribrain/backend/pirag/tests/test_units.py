
from pirag.guards.unit_guard import units_consistent
def test_units():
    assert units_consistent("Temperature is 300 K and pressure 1 atm")
    assert not units_consistent("Value is 10 xyzUnit")
    print("Units guard OK")
