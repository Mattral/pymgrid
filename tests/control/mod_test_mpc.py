import numpy as np
from tests.helpers.test_case import TestCase
from tests.helpers.modular_microgrid import get_modular_microgrid
from pymgrid.algos import ModelPredictiveControl
from pymgrid.modules import RenewableModule, LoadModule

class TestMPC(TestCase):
    
    def setUp(self):
        # Constants for repeated use across tests
        self.max_steps = 10
        self.pv_const = 50
        self.load_const = 60
        self.pv_time_series = np.full(100, self.pv_const)
        self.load_time_series = np.full(100, self.load_const)

    def create_microgrid(self, remove_modules, additional_modules):
        return get_modular_microgrid(remove_modules=remove_modules, additional_modules=additional_modules)

    def test_init(self):
        microgrid = get_modular_microgrid()
        mpc = ModelPredictiveControl(microgrid)
        self.assertTrue(mpc.is_modular)
        self.assertEqual(mpc.horizon, 1)

    def test_run_with_load_pv_battery_grid(self):
        pv = RenewableModule(time_series=self.pv_time_series)
        load = LoadModule(time_series=self.load_time_series)
        microgrid = self.create_microgrid(
            remove_modules=["renewable", "load", "genset"], 
            additional_modules=[pv, load]
        )

        mpc = ModelPredictiveControl(microgrid)
        mpc_output = mpc.run(max_steps=self.max_steps)
        
        # Assertions for output shape and values
        self.assertEqual(mpc_output.shape[0], self.max_steps)
        total_output = (mpc_output[("grid", 0, "grid_import")].values +
                        mpc_output[("battery", 0, "discharge_amount")].values +
                        mpc_output[("renewable", 0, "renewable_used")].values)
        self.assertEqual(total_output, [self.load_const] * mpc_output.shape[0])

    def test_run_with_load_pv_battery_genset(self):
        pv = RenewableModule(time_series=self.pv_time_series)
        load = LoadModule(time_series=self.load_time_series)
        microgrid = self.create_microgrid(
            remove_modules=["renewable", "load", "grid"], 
            additional_modules=[pv, load]
        )

        mpc = ModelPredictiveControl(microgrid)
        mpc_output = mpc.run(max_steps=self.max_steps)
        
        # Assertions for output shape and values
        self.assertEqual(mpc_output.shape[0], self.max_steps)
        self.assertEqual(mpc_output[("load", 0, "load_met")].values, [self.load_const] * mpc_output.shape[0])
        genset_output = (mpc_output[("genset", 0, "genset_production")].values +
                         mpc_output[("battery", 0, "discharge_amount")].values)
        self.assertEqual(genset_output, [10.] * mpc_output.shape[0])

    def test_run_twice_with_load_pv_battery_genset(self):
        pv = RenewableModule(time_series=self.pv_time_series)
        load = LoadModule(time_series=self.load_time_series)
        microgrid = self.create_microgrid(
            remove_modules=["renewable", "load", "grid"], 
            additional_modules=[pv, load]
        )

        mpc = ModelPredictiveControl(microgrid)

        # First run
        mpc_output = mpc.run(max_steps=self.max_steps)
        self.assertEqual(mpc_output.shape[0], self.max_steps)
        self.assertEqual(mpc_output[("load", 0, "load_met")].values, [self.load_const] * mpc_output.shape[0])
        genset_output = (mpc_output[("genset", 0, "genset_production")].values +
                         mpc_output[("battery", 0, "discharge_amount")].values)
        self.assertEqual(genset_output, [10.] * mpc_output.shape[0])

        # Second run
        mpc_output = mpc.run(max_steps=self.max_steps)
        self.assertEqual(mpc_output.shape[0], self.max_steps)
        self.assertEqual(mpc_output[("load", 0, "load_met")].values, [self.load_const] * mpc_output.shape[0])
        self.assertEqual(mpc_output[("genset", 0, "genset_production")].values, [10.] * mpc_output.shape[0])

    def test_run_with_load_pv_battery_grid_different_names(self):
        pv = RenewableModule(time_series=self.pv_time_series)
        load = LoadModule(time_series=self.load_time_series)
        microgrid = self.create_microgrid(
            remove_modules=["renewable", "load", "genset"],
            additional_modules=[("pv_with_name", pv), ("load_with_name", load)]
        )

        mpc = ModelPredictiveControl(microgrid)
        mpc_output = mpc.run(max_steps=self.max_steps)

        # Assertions for output shape and values
        self.assertEqual(mpc_output.shape[0], self.max_steps)
        self.assertEqual(mpc_output[("load_with_name", 0, "load_met")].values, [self.load_const] * mpc_output.shape[0])
        total_output = (mpc_output[("grid", 0, "grid_import")].values +
                        mpc_output[("battery", 0, "discharge_amount")].values +
                        mpc_output[("pv_with_name", 0, "renewable_used")].values)
        self.assertEqual(total_output, [self.load_const] * mpc_output.shape[0])
