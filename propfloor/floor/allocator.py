"""Assign validated, mutually low-correlation strategies to floor accounts."""
from __future__ import annotations

import pandas as pd


def correlation_matrix(returns: dict[str, dict[str, float]]) -> pd.DataFrame:
    df = pd.DataFrame(returns).fillna(0.0)
    if df.empty:
        return pd.DataFrame()
    return df.corr().fillna(0.0)


def select_diversified(candidates: list[dict], returns: dict[str, dict[str, float]], k: int,
                       max_corr: float = 0.3, score_key: str = "p_pass_oos") -> list[str]:
    """Greedy: best score first, skip anything correlated > max_corr with what is already chosen."""
    corr = correlation_matrix({c["id"]: returns.get(c["id"], {}) for c in candidates})
    chosen: list[str] = []
    for c in sorted(candidates, key=lambda r: -(r.get(score_key) or 0)):
        sid = c["id"]
        if any(abs(corr.loc[sid, o]) > max_corr for o in chosen if sid in corr.index and o in corr.index):
            continue
        chosen.append(sid)
        if len(chosen) >= k:
            break
    return chosen


def allocate(candidates: list[dict], returns: dict[str, dict[str, float]], n_accounts: int, per_account: int = 2,
             max_corr: float = 0.3) -> list[list[str]]:
    """Each account gets a diversified basket; baskets rotate through the pool so accounts differ."""
    pool = select_diversified(candidates, returns, k=max(per_account * n_accounts, per_account), max_corr=max_corr)
    if not pool:
        return [[] for _ in range(n_accounts)]
    out = []
    for i in range(n_accounts):
        out.append([pool[(i * per_account + j) % len(pool)] for j in range(min(per_account, len(pool)))])
    return out
