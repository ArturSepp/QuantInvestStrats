"""Inspect annualisation factors across supported frequency aliases."""

from enum import Enum

from qis.utils.annualisation import get_annualization_factor


FREQUENCIES = [
    # Intraday
    "1M", "5M", "15M", "15T", "h", "H",
    # Daily
    "D", "B", "C",
    # Weekly
    "W", "WE", "W-MON", "W-FRI", "WE-WED",
    "2W", "2WE", "2W-FRI", "SM",
    "3W", "3W-MON",
    "4W", "4W-FRI",
    # Monthly
    "M", "ME", "1M", "MS", "BM", "BMS",
    "2M", "2ME", "2BM",
    # Quarterly
    "Q", "QE", "QS", "BQ", "BQS",
    "Q-DEC", "QE-DEC", "QE-JAN", "QE-FEB",
    "2Q", "2QE", "2BQ",
    "3Q", "3QE",
    # Annual
    "Y", "YE", "A", "YS", "AS", "BA", "BAS",
]


class Locals(Enum):
    """Available annualisation diagnostics."""

    PRINT_FACTORS = 1


def run_local(local: Locals) -> None:
    """Print annualisation factors for the selected diagnostic."""
    if local != Locals.PRINT_FACTORS:
        raise ValueError(f"unsupported local diagnostic: {local}")

    print("Frequency → Annualization Factor")
    print("=" * 40)
    for frequency in FREQUENCIES:
        annualization_factor = get_annualization_factor(frequency)
        print(f"{frequency:12s} → {annualization_factor:8.2f}")

    print("\n" + "=" * 40)
    print("Calendar vs Trading Days (B frequency)")
    print("=" * 40)
    print(f"B (trading):  {get_annualization_factor('B', is_calendar=False):.2f}")
    print(f"B (calendar): {get_annualization_factor('B', is_calendar=True):.2f}")
    print(f"D (always):   {get_annualization_factor('D', is_calendar=False):.2f}")


if __name__ == "__main__":
    run_local(local=Locals.PRINT_FACTORS)
