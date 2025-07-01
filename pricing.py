import pandas as pd

def price_storage_contract(
    injection_dates,
    withdrawal_dates,
    price_series,       # pandas Series or dict: {date: price}
    inject_rate,        # units per day
    withdraw_rate,      # units per day
    max_volume,         # units
    storage_costs       # per day or per unit
):
    """
    injection_dates: list of dates when gas is injected
    withdrawal_dates: list of dates when gas is withdrawn
    price_series: dict or pd.Series mapping dates to prices
    inject_rate: max gas injected per date
    withdraw_rate: max gas withdrawn per date
    max_volume: storage capacity
    storage_costs: daily cost per unit stored (or total, adjust as needed)
    """
    # Track storage level over time
    storage = 0
    cashflow = 0
    storage_log = []

    # Combine all relevant dates, sorted
    all_dates = sorted(set(injection_dates + withdrawal_dates))
    
    for date in all_dates:
        price = price_series[date] if isinstance(price_series, dict) else price_series.loc[date]
        
        # Inject gas if it's an injection date
        if date in injection_dates:
            inject_amount = min(inject_rate, max_volume - storage)
            storage += inject_amount
            cashflow -= inject_amount * price  # Buying gas
        # Withdraw gas if it's a withdrawal date
        if date in withdrawal_dates:
            withdraw_amount = min(withdraw_rate, storage)
            storage -= withdraw_amount
            cashflow += withdraw_amount * price  # Selling gas
        
        # Apply storage costs (adjust for your definition: per day, per unit, etc.)
        daily_cost = storage * storage_costs
        cashflow -= daily_cost
        storage_log.append((date, storage, cashflow))
    
    return cashflow, storage_log  # Optionally, also return storage over time

# --- Test Example ---
price_data = {
    '2024-07-01': 3.0,
    '2024-07-02': 3.2,
    '2024-07-03': 3.3,
    '2024-07-04': 3.1,
    '2024-07-05': 3.5,
}
injection = ['2024-07-01', '2024-07-03']
withdrawal = ['2024-07-05']
inject_rate = 1000
withdraw_rate = 1000
max_volume = 2000
storage_costs = 0.02  # per unit per day

contract_value, log = price_storage_contract(
    injection, withdrawal, price_data, inject_rate, withdraw_rate, max_volume, storage_costs
)

print("Contract value:", contract_value)
print("Storage/cash log:", log)
