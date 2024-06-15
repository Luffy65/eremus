import pkgutil
try:
    import comet_ml
    has_cml = True
except:
    has_cml = False

__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    if is_pkg or "." in module_name:
        continue
    __all__.append(module_name)
    _module = loader.find_module(module_name).load_module(module_name)
    globals()[module_name] = _module
