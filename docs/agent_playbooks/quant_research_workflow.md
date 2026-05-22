# Quant Research Agent Playbook

Use this playbook for research-heavy tasks such as adding a new signal, testing a
microstructure hypothesis, or reviewing a paper-to-code implementation.

## Workflow

1. State the hypothesis in market terms.
2. Identify the exact data needed and its timestamp convention.
3. Define train/validation/test boundaries before looking at results.
4. Build features using only information available at the decision timestamp.
5. Define the markout or PnL formula with signs, units, and fees/rebates.
6. Run a small smoke sample before full data.
7. Report results by symbol, date, regime, and parameter setting.
8. Write down failure modes and whether the result survives out-of-sample checks.

## HFT Checks

- Is `side` maker side, taker side, or order side?
- Are timestamps exchange time, local receive time, or normalized UTC?
- Is latency modeled or at least acknowledged?
- Are fills possible at the quoted price under the available market data?
- Is turnover large enough for the result to matter?
- Is the signal still useful after fees, rebates, and inventory constraints?
