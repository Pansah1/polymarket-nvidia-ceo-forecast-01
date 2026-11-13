# polymarket-nvidia-ceo-forecast-01

Entry for the make.fun `polymarket-nvidia-ceo-forecast-01` competition:
forecast Polymarket's "Next CEO of NVIDIA?" Yes prices, minute by minute,
four hours ahead, as five quantiles (5/25/50/75/95%) per market per minute.

## Method

The house baseline's persistence forecast -- the price stays where it was,
with a spread widening like a random walk's -- with the spread tuned on the
practice cases instead of the baseline's fixed 0.0025. Backtesting on the
six practice cases (cut 240 minutes before each checkpoint, scored against
the prices that followed) showed two things:

- The median belongs at the last price. Pulling it toward a longer-term
  mean (mean reversion) made the loss much worse at every weight tried.
- Per-market volatility estimates hurt: recent wiggles do not predict how
  far a market will move next. One spread, tuned on all the practice cases
  together, beats the baseline and every adaptive variant.

So the forecast is q_tau(h) = clamp(x + z_tau * 0.0018 * sqrt(h), 0, 1),
with x the market's last price and z_tau the normal quantile at each level.
Backtest pinball loss: 0.418 vs the baseline's 0.430 on the same windows.

## License

MIT. See LICENSE.
