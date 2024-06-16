import pkgutil
import importlib
import importlib.util







__all__ = []
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__):
    if is_pkg or "." in module_name:
        continue
    print(f"Importing module: {module_name}")  # Debug print
    __all__.append(module_name)
    try:
        spec = importlib.util.find_spec(f'.{module_name}', package=__name__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        globals()[module_name] = module
    except Exception as e:
        print(f"Error importing module {module_name}: {e}")  # Debug print