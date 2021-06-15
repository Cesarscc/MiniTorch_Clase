## Task 0.4
## Modules


class Module:
    """
    Attributes:
        _modules (dict of name x :class:`Module`): Storage of the child modules
        _parameters (dict of name x :class:`Parameter`): Storage of the module's parameters
        mode (string): Mode of operation, can be {"train", "eval"}.
    """

    def __init__(self):
        self._modules = {}
        self._parameters = {}
        self.mode = "train"

    def modules(self):
        "Return the child modules of this module."
        return self.__dict__["_modules"].values()

    def train(self):
        "Set the mode of this module and all descendent modules to `train`."
        self.mode = "train"
        if len(self._modules) != 0:
            for key in self._modules:
                self._modules[key].train()
        # raise NotImplementedError('Need to implement for Task 0.4')

    def eval(self):
        "Set the mode of this module and all descendent modules to `eval`."
        self.mode = "eval"
        if len(self._modules) != 0:
            for key in self._modules:
                self._modules[key].eval()
        # raise NotImplementedError('Need to implement for Task 0.4')

    def named_parameters(self):
        """
        Collect all the parameters of this module and its descendents.
        Returns:
            dict: Each name (key) and :class:`Parameter` (value) under this module.
        """
        if len(self._modules) == 0:
            newDict = {}
            for param in self._parameters:
                newDict[param] = self._parameters[param]
            return newDict
        else:
            thisParam = {}
            for key in self._parameters:
                thisParam[key] = self._parameters[key]
            for key in self._modules:
                params = self._modules[key].named_parameters()
                for name in params:
                    thisParam[key + "." + name] = params[name]
            return thisParam
        # raise NotImplementedError('Need to implement for Task 0.4')

    def parameters(self):
        return self.named_parameters().values()

    def add_parameter(self, k, v):
        """
        Manually add a parameter. Useful helper for scalar parameters.
        Args:
            k (str): Local name of the parameter.
            v (value): Value for the parameter.
        Returns:
            Parameter: Newly created parameter.
        """
        val = Parameter(v)
        self.__dict__["_parameters"][k] = val
        return val

    def __setattr__(self, key, val):
        if isinstance(val, Parameter):
            self.__dict__["_parameters"][key] = val
        elif isinstance(val, Module):
            self.__dict__["_modules"][key] = val
        else:
            super().__setattr__(key, val)

    def __getattr__(self, key):
        if key in self.__dict__["_parameters"]:
            return self.__dict__["_parameters"][key]

        if key in self.__dict__["_modules"]:
            return self.__dict__["_modules"][key]

        return self.__getattribute__(key)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self):
        assert False, "Not Implemented"

    def __repr__(self):
        def _addindent(s_, numSpaces):
            u = s_.split("\n")
            if len(u) == 1:
                return s_
            first = u.pop(0)
            u = [(numSpaces * " ") + line for line in u]
            u = "\n".join(u)
            u = first + "\n" + u
            return u

        child_lines = []

        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append("(" + key + "): " + mod_str)
        lines = child_lines

        main_str = self.__class__.__name__ + "("
        if lines:
            # simple one-liner info, which most builtin Modules will use
            main_str += "\n  " + "\n  ".join(lines) + "\n"

        main_str += ")"
        return main_str


class Parameter:
    """
    A Parameter is a special container stored in a :class:`Module`.
    It is designed to hold a :class:`Variable`, but we allow it to hold
    any value for testing.
    """

    def __init__(self, x=None):
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def update(self, x):
        "Update the parameter value."
        self.value = x
        if hasattr(x, "requires_grad_"):
            self.value.requires_grad_(True)

    def __repr__(self):
        return repr(self.value)