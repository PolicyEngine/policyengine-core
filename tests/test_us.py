def test_policyengine_us_microsimulation_runs():
    from policyengine_us import Microsimulation

    baseline = Microsimulation()
    baseline.subsample(100)

    baseline.calculate("household_net_income")
