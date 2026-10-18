"""Tax schedules and announcement-aware scheduled changes."""

import pandas as pd
import pytest

from gasprix import taxes


def test_federal_excise_suspension_and_return():
    assert taxes.components('QC', 'montreal', '2026-04-19')['federal_excise'] == 10.0
    assert taxes.components('QC', 'montreal', '2026-04-20')['federal_excise'] == 0.0
    assert taxes.components('QC', 'montreal', '2027-02-15')['federal_excise'] == 5.0
    assert taxes.components('QC', 'montreal', '2027-04-01')['federal_excise'] == 10.0


def test_carbon_charge_removed_outside_quebec():
    assert taxes.components('ON', 'toronto', '2025-03-31')['carbon'] == pytest.approx(17.61)
    assert taxes.components('ON', 'toronto', '2025-04-01')['carbon'] == 0.0
    assert taxes.components('QC', 'montreal', '2024-06-01')['carbon'] == 0.0


def test_montreal_transit_levy_and_sales_tax():
    c = taxes.components('QC', 'montreal', '2026-10-01')
    assert c['provincial'] + c['municipal'] == pytest.approx(22.2)
    assert c['sales_rate'] == pytest.approx(0.14975)


@pytest.mark.parametrize('origin, expected', [
    ('2026-03-30', 0.0),      # suspension not yet announced
    ('2026-04-15', -11.5),    # announced April 14, starts April 20
    ('2026-08-20', 11.5),     # the (later cancelled) September 8 return was expected
    ('2026-09-03', 0.0),      # extension announced September 2
])
def test_scheduled_change_uses_only_announced_information(origin, expected):
    change = taxes.scheduled_change('QC', 'montreal', pd.DatetimeIndex([origin]), 28)
    assert change[0] == pytest.approx(expected, abs=0.1)


def test_alberta_relief_program_october_2026():
    assert taxes.components('AB', 'calgary', '2026-09-30')['provincial'] == 13.0
    assert taxes.components('AB', 'calgary', '2026-10-01')['provincial'] == 0.0
    before = taxes.scheduled_change('AB', 'calgary', pd.DatetimeIndex(['2026-09-15']), 28)[0]
    after = taxes.scheduled_change('AB', 'calgary', pd.DatetimeIndex(['2026-09-23']), 28)[0]
    assert before == pytest.approx(0.0) and after == pytest.approx(-13.65)


def test_breakdown_sums_to_price():
    b = taxes.breakdown(190.0, 'ON', 'toronto', '2026-10-01')
    assert sum(b.values()) == pytest.approx(190.0)
