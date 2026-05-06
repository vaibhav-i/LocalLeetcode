def max_profit(prices):
    best_buy = float("inf")
    best_profit = 0
    for price in prices:
        best_buy = min(best_buy, price)
        best_profit = max(best_profit, price - best_buy)
    return best_profit
