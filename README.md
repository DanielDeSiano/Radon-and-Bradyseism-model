# Radon and Bradyseism model

A physically based model of ²²²Rn and ²²⁰Rn transport in the unsaturated zone,
written to answer one question: can soil radon be used to monitor volcanic
unrest and bradyseism? I live on Ischia, in the same volcanic district as
Campi Flegrei, so for me it is not an abstract question.

This repository holds the model: one file, 8,543 lines of Python. The
preprint, the verification suite and the blind-validation files are in the
Zenodo deposit: https://doi.org/10.5281/zenodo.22755472

## Use

```bash
pip install -r requirements.txt
```

```python
import importlib.util
spec = importlib.util.spec_from_file_location("vrm", "volcanic_radon_model_v11_finale.py")
vrm = importlib.util.module_from_spec(spec); spec.loader.exec_module(vrm)

m = vrm.build_model("tuff_unwelded")   # or granite, limestone_karst, scoria_basaltic, ...
res = m.simulate_fast(days=3.0)        # operator-splitting solver
sol = m.simulate(days=3.0)             # fully coupled implicit (BDF) reference
```

Twenty lithotypes are parameterised (`GEOLITHOTYPES`).

## License and citation

MIT. To cite, use the Zenodo DOI above or `CITATION.cff`.
