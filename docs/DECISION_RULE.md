# Pre-Committed Decision Rule: Logistic Regression (LR) vs. Random Forest (RF)

*Pre-commitment Date: 15 November 2026*  
*Project: Project Triathlon*  
*Status: Locked & Pre-Committed prior to Experiment Execution E0–E5*

---

- $d = \text{mean regret}(\text{RF}) - \text{mean regret}(\text{LR})$ in dB, TUNED arms, LOL-v1 grouped CV (negative = RF better), 95% CI $[d_{lo}, d_{hi}]$; $\delta = 0.10\text{ dB}$.
- **RF_WINS** iff:
  1. $d_{hi} < -\delta$;
  2. $d < 0$ on BOTH external evaluations (LOL-v2-real Test and the reverse direction);
  3. in E2(b) $d_{hi} < 0$ at threshold 18;
  4. all E5 controls pass.
- **RF_CV_ONLY**: $d_{hi} < 0$ but (2), (3) or (4) fails; call it probable overfitting or leakage.
- **LR_WINS**: $d_{lo} > 0$. **TIE**: anything else ("no detectable difference").
- Deploy RF only for **RF_WINS**; in every other case deploy LR (the burden of proof is on the more complex model). Deploy tuned hyper-parameters only if E1 shows tuned minus default regret CI entirely below 0; otherwise the defaults.
- If another learner beats both by the same criteria, do not deploy it: report it prominently and flag it in the RESULT CARD for my decision. Report everything, whatever the outcome.
