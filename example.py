from pyfly.pyfly import PyFly
from pyfly.pid_controller import PIDController
import numpy as np

sim = PyFly("/home/bitdrones/fixed_wing/pyfly/pyfly/pyfly_config.json", "/home/bitdrones/fixed_wing/pyfly/pyfly/x8_param.mat")
sim.seed(0)

sim.reset(state={"roll": -0.5, "pitch": 0.15})

pid = PIDController(sim.dt)
pid.set_reference(phi=0.2, theta=0, va=22)

for step_i in range(500):
    phi = sim.state["roll"].value
    theta = sim.state["pitch"].value
    Va = sim.state["Va"].value
    omega = [sim.state["omega_p"].value, sim.state["omega_q"].value, sim.state["omega_r"].value]

    action = pid.get_action(phi, theta, Va, omega)
    success, step_info = sim.step(action)
    f = sim.simplified_forces([ 0.96535389, -0.24855198,  0.06601354,  0.04422673], np.eye(3),
                           [0.6520966716678179, -0.3881707606010084, 0.5485362513953406]
                           , np.array([19.94188579, -3.19730311, -3.45837158]))
    print(f)
    input()

    if not success:
        break

sim.render(block=True)