from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PlanResult:
    steps: List[str]
    buy_dollars: Dict[str, float]
    sell_dollars: Dict[str, float]
    final_values: Dict[str, float]
    final_weights: Dict[str, float]

def rebalance_execution_plan(
    current_values: Dict[str, float],
    target_weights: Dict[str, float],
    cash: float = None,                 # optional; if None, will use current_values.get("CASH", 0)
    transaction_cost_pct: float = 0.0,  # e.g., 0.001 = 0.1% applied to buy/sell notionals
    min_instruction_threshold: float = 1e-6  # suppress tiny noise in output
) -> PlanResult:
    """
    Build a dollars-only, step-by-step execution plan to reach target weights.
    - current_values: dict of ticker -> current $ value. Include "CASH" if you track it as a position.
    - target_weights: dict of ticker -> target weight (must sum to 1.0). May include "CASH".
    - cash: optional explicit cash balance. If None, pulled from current_values.get("CASH", 0).
    - transaction_cost_pct: proportional fee on trade notional. Buys cost $*(1+fee); sells yield $*(1-fee).
    - Returns: PlanResult with chronological 'steps' and summary dicts.
    """

    fee = float(transaction_cost_pct)
    if fee < 0 or fee >= 1:
        raise ValueError("transaction_cost_pct must be in [0, 1).")

    # Normalize keys (treat tickers not in target as target weight 0).
    tickers = sorted(set(current_values.keys()).union(target_weights.keys()))
    # Pull cash from current_values if not explicitly provided.
    if cash is None:
        cash = float(current_values.get("CASH", 0.0))

    # Build current asset values excluding cash key for asset math.
    cur = {}
    for t in tickers:
        if t == "CASH": 
            continue
        cur[t] = float(current_values.get(t, 0.0))
    # Ensure every ticker in target has a number (0 if missing).
    target_w = {t: float(target_weights.get(t, 0.0)) for t in tickers}

    # Validate target weights sum
    wsum = sum(target_w.values())
    if abs(wsum - 1.0) > 1e-9:
        raise ValueError(f"Target weights must sum to 1.0; got {wsum:.6f}")

    # Totals and targets (include cash)
    starting_total = sum(cur.values()) + cash
    target_dollars = {t: target_w.get(t, 0.0) * starting_total for t in tickers}
    target_cash = target_dollars.get("CASH", 0.0)

    # Deltas by asset (positive = buy needed, negative = sell needed)
    deltas = {}
    for t in tickers:
        if t == "CASH":
            continue
        deltas[t] = target_dollars.get(t, 0.0) - cur.get(t, 0.0)

    # Split under/over weights (ignore negligible noise)
    to_buy = {t: max(0.0, d) for t, d in deltas.items() if d > min_instruction_threshold}
    to_sell = {t: max(0.0, -d) for t, d in deltas.items() if d < -min_instruction_threshold}

    steps: List[str] = []
    buy_used = {t: 0.0 for t in deltas}
    sell_used = {t: 0.0 for t in deltas}

    # Helper to compute remaining needs
    def remaining_buy_total() -> float:
        return sum(to_buy.values())

    # 1) Use existing cash to fund buys first (cheapest path).
    if remaining_buy_total() > 0 and cash > min_instruction_threshold:
        # Buy biggest deficits first
        for t in sorted(to_buy, key=lambda x: -to_buy[x]):
            if to_buy[t] <= min_instruction_threshold or cash <= min_instruction_threshold:
                continue
            # To increase position by X, cash outlay must cover fees: outlay = X * (1 + fee)
            needed = to_buy[t] * (1.0 + fee)
            use_outlay = min(needed, cash)
            # The net position increase achieved with this outlay:
            net_increase = use_outlay / (1.0 + fee)
            if net_increase <= min_instruction_threshold:
                continue
            to_buy[t] -= net_increase
            buy_used[t] += net_increase
            cash -= use_outlay
            steps.append(f"Buy ${net_increase:.2f} of {t} using available CASH (spend ${use_outlay:.2f}")

    # 2) If still need funding for buys and/or need to raise target cash, sell overweights.
    # Desired change in cash: positive means we must end with *more* cash.
    delta_cash = target_cash - cash  # we compare against current cash *after step 1 purchases*
    # Proceeds needed to finish remaining buys:
    proceeds_needed_for_buys = 0.0
    if remaining_buy_total() > min_instruction_threshold:
        # For remaining net buy increase B, we must spend B * (1 + fee),
        # which must come from sell proceeds. Each $ of sell notional yields (1 - fee) cash.
        B = remaining_buy_total()
        proceeds_needed_for_buys = B * (1.0 + fee)

    # If we must *increase* cash by delta_cash > 0, that must also be covered by net proceeds.
    # Total cash proceeds we need to raise from sells:
    total_cash_proceeds_needed = max(0.0, proceeds_needed_for_buys + max(0.0, delta_cash))

    # Execute sells from biggest overweights
    if total_cash_proceeds_needed > min_instruction_threshold:
        for t in sorted(to_sell, key=lambda x: -to_sell[x]):
            if total_cash_proceeds_needed <= min_instruction_threshold:
                break
            if to_sell[t] <= min_instruction_threshold:
                continue
            # Selling S notional yields S*(1-fee) cash
            max_sell_notional = to_sell[t]
            max_cash_from_t = max_sell_notional * (1.0 - fee)
            sell_cash = min(max_cash_from_t, total_cash_proceeds_needed)
            sell_notional = sell_cash / (1.0 - fee)

            # Apply the sell
            to_sell[t] -= sell_notional
            sell_used[t] += sell_notional
            cash += sell_cash
            total_cash_proceeds_needed -= sell_cash
            steps.append(f"Sell ${sell_notional:.2f} of {t} (receive ${sell_cash:.2f}")

            # Immediately allocate to any remaining buys (largest deficits first)
            # Note: allocation uses cash; for each $X increase we must spend X*(1+fee)
            if remaining_buy_total() > min_instruction_threshold and cash > min_instruction_threshold:
                for b in sorted(to_buy, key=lambda x: -to_buy[x]):
                    if to_buy[b] <= min_instruction_threshold or cash <= min_instruction_threshold:
                        continue
                    needed_outlay = to_buy[b] * (1.0 + fee)
                    use_outlay = min(needed_outlay, cash)
                    net_inc = use_outlay / (1.0 + fee)
                    if net_inc <= min_instruction_threshold:
                        continue
                    to_buy[b] -= net_inc
                    buy_used[b] += net_inc
                    cash -= use_outlay
                    steps.append(f" → With ${use_outlay:.2f} cash, buy ${net_inc:.2f} of {b}")

    # 3) If buys still remain, we couldn't raise enough from overweights.
    if remaining_buy_total() > min_instruction_threshold:
        steps.append("Warning: Not enough funding from overweights/CASH to complete all buys.")

    # 4) If we have excess cash above target (delta_cash < 0), we should spend it on any tiny residual buys.
    # Otherwise we leave it as extra cash (minor drift).
    delta_cash_final = target_cash - cash
    if delta_cash_final < -min_instruction_threshold and remaining_buy_total() <= min_instruction_threshold:
        steps.append(f"Note: Ending CASH ${cash:.2f} exceeds target by ${-delta_cash_final:.2f}. (Small drift retained.)")

    # Build final values and weights (approx, post-fee)
    final_values = {}
    for t in deltas:
        final_values[t] = cur.get(t, 0.0) + buy_used.get(t, 0.0) - sell_used.get(t, 0.0)
    final_cash = cash
    final_total = final_cash + sum(final_values.values())
    final_values["CASH"] = final_cash

    final_weights = {t: (final_values[t] / final_total if final_total > 0 else 0.0) for t in final_values}

    # Summaries of buy/sell notionals by ticker (suppress micros)
    buy_summary = {t: v for t, v in buy_used.items() if v > min_instruction_threshold}
    sell_summary = {t: v for t, v in sell_used.items() if v > min_instruction_threshold}

    return PlanResult(
        steps=steps,
        buy_dollars=buy_summary,
        sell_dollars=sell_summary,
        final_values=final_values,
        final_weights=final_weights,
    )


# -------------------------
# Example usage
if __name__ == "__main__":
    current = {
        "GOOGL": 4709.0,
        "AMZN": 3604.0,
        "UNH": 3105.0,
        "ASML": 3000.0,
        "HLT": 2286.0,
        "BKNG": 1976.0,
        "UBER": 1960.0,
        "HOOD": 1753.0,
        "AAAU": 1346.0,
        "V": 347.0,
        "CASH": 1544,
    }

    # Replace the values below with your desired target percentages.
    # NOTE: All numbers must sum to 1.0
    target = {
        "GOOGL": 0.15,
        "AMZN":  0.14,
        "UNH":   0.12,
        "ASML":  0.11,
        "HLT":   0.09,
        "BKNG":  0.13,
        "UBER":  0.10,
        "HOOD":  0.04,
        "AAAU":  0.05,
        "V":     0.00,
        "CASH":  0.07,
    }

    plan = rebalance_execution_plan(
        current_values=current,
        target_weights=target,
        cash=None,                   # pull from current["CASH"]
        transaction_cost_pct=0.0     # set e.g. 0.001 for 0.1%
    )

    print("=== STEP-BY-STEP PLAN ===")
    for s in plan.steps:
        print(s)

    print("\n=== SUMMARY ===")
    print("Buy ($):", plan.buy_dollars)
    print("Sell ($):", plan.sell_dollars)
    print("Final Values ($):", {k: round(v, 2) for k, v in plan.final_values.items()})
    print("Final Weights:", {k: round(v, 4) for k, v in plan.final_weights.items()})
