

class ParamDict:
    """Hold a parameter dict for easy parameter injection
    """
    def __init__(self, params) -> None:
        self._params = ParamDict._augment_params(params)

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

        return par

    def copy(self):
        # Save parsing
        x = ParamDict({})
        x._params = self._params.copy()
        return x

    def __getitem__(self, key):
        return self._params[key]

    def __setitem__(self, key, value):
        try:
            method, param_name = key.split(':')
            meth_dict = self._params.setdefault(method, dict())
            meth_dict[param_name] = value
        except:
            if not isinstance(value, dict):
                raise ValueError

            self._params[key] = value.copy()
