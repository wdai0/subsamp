"""Run with python -I after wheel installation; no pytest source-path injection."""
from pathlib import Path
import importlib.metadata
import json

import numpy as np
import subsampwinner
from subsampwinner import fit_least_squares, subsamp

package = Path(subsampwinner.__file__).resolve()
assert 'site-packages' in package.parts, f'Expected installed package, got {package}'
rng = np.random.RandomState(301)
X = rng.normal(size=(40, 4))
y = 1 + 2 * X[:, 0] + rng.normal(size=40)
fit = fit_least_squares(X, y)
assert np.isfinite(fit.p_values).all()
swa = subsamp(2, 12, seed=302)
swa.fit(X, y)
assert len(swa.semifinalists) == 2 and np.isfinite(swa.final_model.pvalues).all()
print(json.dumps({'package': str(package), 'versions': {
    name: importlib.metadata.version(name) for name in
    ('numpy', 'scipy', 'statsmodels', 'pandas', 'patsy')}, 'status': 'passed'}, indent=2))
