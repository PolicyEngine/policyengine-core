"""Helpers for tests of branches that share their parent's cached arrays."""

from policyengine_core.simulations import SimulationBuilder

SITUATION = {
    "persons": {
        "a": {"birth": {"ETERNITY": "1950-03-01"}, "salary": {"2017-01": 4000}},
        "b": {"birth": {"ETERNITY": "1984-01-01"}, "salary": {"2017-01": 2500}},
        "c": {"birth": {"ETERNITY": "2010-06-01"}},
        "d": {"birth": {"ETERNITY": "1940-01-01"}, "salary": {"2017-01": 100}},
    },
    "households": {
        "h1": {
            "parents": ["a", "b"],
            "children": ["c"],
            "rent": {"2017-01": 1200},
            "accommodation_size": {"2017-01": 80},
            "housing_occupancy_status": {"2017-01": "tenant"},
        },
        "h2": {
            "parents": ["d"],
            "rent": {"2017-01": 300},
            "accommodation_size": {"2017-01": 40},
            "housing_occupancy_status": {"2017-01": "owner"},
        },
    },
}


def build_simulation(tax_benefit_system):
    return SimulationBuilder().build_from_entities(tax_benefit_system, SITUATION)


def _storages(simulation):
    return {
        name: holder._memory_storage
        for population in simulation.populations.values()
        for name, holder in population._holders.items()
    }


def stored_arrays(simulation):
    """Every (variable, storage key) -> array in a simulation's memory storage."""
    return {
        (name, key): array
        for name, storage in _storages(simulation).items()
        for key, array in storage._arrays.items()
    }


def shared_keys(simulation):
    """The (variable, storage key) entries a simulation has not copied yet."""
    return {
        (name, key)
        for name, storage in _storages(simulation).items()
        for key in storage._shared
    }
