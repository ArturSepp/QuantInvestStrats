
from qis.utils.annualisation import get_annualization_factor

freqs = [
    # Intraday
    '1M', '5M', '15M', '15T', 'h', 'H',
    # Daily
    'D', 'B', 'C',
    # Weekly
    'W', 'WE', 'W-MON', 'W-FRI', 'WE-WED',
    '2W', '2WE', '2W-FRI', 'SM',
    '3W', '3W-MON',
    '4W', '4W-FRI',
    # Monthly
    'M', 'ME', '1M', 'MS', 'BM', 'BMS',
    '2M', '2ME', '2BM',
    # Quarterly
    'Q', 'QE', 'QS', 'BQ', 'BQS',
    'Q-DEC', 'QE-DEC', 'QE-JAN', 'QE-FEB',
    '2Q', '2QE', '2BQ',
    '3Q', '3QE',
    # Annual
    'Y', 'YE', 'A', 'YS', 'AS', 'BA', 'BAS',
]

print("Frequency → Annualization Factor")
print("=" * 40)
for freq in freqs:
    an_factor = get_annualization_factor(freq)
    print(f"{freq:12s} → {an_factor:8.2f}")

# Test with is_calendar parameter
print("\n" + "=" * 40)
print("Calendar vs Trading Days (B frequency)")
print("=" * 40)
print(f"B (trading):  {get_annualization_factor('B', is_calendar=False):.2f}")
print(f"B (calendar): {get_annualization_factor('B', is_calendar=True):.2f}")
print(f"D (always):   {get_annualization_factor('D', is_calendar=False):.2f}")
