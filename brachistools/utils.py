

class ParamDict:
    """Hold a parameter dict for easy parameter injection
    """
    def __init__(self, params) -> None:
        self._params = self._augment_params(params)

    @classmethod
    def _augment_params(cls, params: dict):
        par = params.copy()
        for k, v in params.items():
            try:
                method, param_name = k.split(sep=':')
                meth_dict = par.setdefault(method, dict())
                meth_dict[param_name] = v
            except:
                pass

            if isinstance(v, dict):
                for k2, v2 in v.items():
                    par[f"{k}:{k2}"] = v2

        par.update(params)
        return par

    def __getattr__(self, key):
        return self._params[key]
