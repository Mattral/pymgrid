import numpy as np
from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, GensetModule, GridModule, LoadModule, RenewableModule

def get_modular_microgrid(remove_modules=(),
                          retain_only=None,
                          additional_modules=None,
                          add_unbalanced_module=True,
                          timeseries_length=100,
                          modules_only=False):
    # Initialize modules with consistent parameter definitions
    modules = {
        'genset': GensetModule(
            running_min_production=10,
            running_max_production=50,
            genset_cost=0.5
        ),
        'battery': BatteryModule(
            min_capacity=0,
            max_capacity=100,
            max_charge=50,
            max_discharge=50,
            efficiency=1.0,
            init_soc=0.5
        ),
        'renewable': RenewableModule(
            time_series=np.full(timeseries_length, 50)
        ),
        'load': LoadModule(
            time_series=np.full(timeseries_length, 60)
        ),
        'grid': GridModule(
            max_import=100,
            max_export=0,
            time_series=np.ones((timeseries_length, 3)),
            raise_errors=True
        )
    }

    # Check for mutually exclusive parameters
    if retain_only is not None:
        if remove_modules:
            raise RuntimeError('Use either "remove_modules" or "retain_only", not both.')
        modules = {k: v for k, v in modules.items() if k in retain_only}
    else:
        # Remove specified modules with error handling for invalid keys
        for module in remove_modules:
            if module in modules:
                del modules[module]
            else:
                raise NameError(f"Module '{module}' not in default modules: {list(modules.keys())}.")

    # Add any additional modules provided
    if additional_modules:
        modules.update({mod.name: mod for mod in additional_modules})

    # Return modules list or Microgrid instance based on modules_only flag
    return list(modules.values()) if modules_only else Microgrid(list(modules.values()), add_unbalanced_module=add_unbalanced_module)
