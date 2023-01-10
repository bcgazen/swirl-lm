# Copyright 2022 The swirl_lm Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding=utf-8
"""A library for solving the momentum equation for velocity.

   #
   # Monitors
   #

   Currently following monitors are also supported:

     MONITOR_velocity_raw_rho-u: It records the raw rho_u component after
       velocity correction, and similarly for the *_rho-{v, w} components.
     MONITOR_velocity_raw_rho-v:
     MONITOR_velocity_raw_rho-w:

   They can be activated by adding the corresponding keys in the
   `helper_var_keys` part of the config file.

   #
   # Density
   #

   Because of the staggering in time, and density is at the same time step as
   scalars, the average density (old & new) is at the same time step as the
   velocity at the new time step.
"""

import functools
from typing import Optional, Sequence, Text

import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import physical_variable_keys_manager
from swirl_lm.boundary_condition import boundary_condition_utils
from swirl_lm.boundary_condition import immersed_boundary_method
from swirl_lm.communication import halo_exchange
from swirl_lm.equations import common
from swirl_lm.equations import utils as eq_utils
from swirl_lm.numerics import convection
from swirl_lm.numerics import diffusion
from swirl_lm.numerics import numerics_pb2
from swirl_lm.numerics import time_integration
from swirl_lm.physics.thermodynamics import thermodynamics_manager
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.physics.thermodynamics import water
from swirl_lm.physics.turbulence import sgs_model
from swirl_lm.utility import common_ops
from swirl_lm.utility import components_debug
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import monitor
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

_GRAVITY = 9.81
# A small number that's used as the threshold for the gravity vector. If the
# absolute value of a gravity component is less than this threshold, it is
# considered as 0 when computing the free slip wall boundary condition.
_G_THRESHOLD = 1e-6

_ConvectionScheme = numerics_pb2.ConvectionScheme
_DiffusionScheme = numerics_pb2.DiffusionScheme

# Density keys.
_KEY_RHO = common.KEY_RHO
# Pressure keys.
_KEY_P = common.KEY_P

# Velocity keys.
_KEY_U = common.KEY_U
_KEY_V = common.KEY_V
_KEY_W = common.KEY_W
_KEYS_VELOCITY = common.KEYS_VELOCITY

# Momentum keys.
_KEY_RHO_U = common.KEY_RHO_U
_KEY_RHO_V = common.KEY_RHO_V
_KEY_RHO_W = common.KEY_RHO_W
_KEYS_MOMENTUM = common.KEYS_MOMENTUM


class Velocity(object):
  """A library for advancing velocity to the next time step."""

  def __init__(
      self,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      params: parameters_lib.SwirlLMParameters,
      thermodynamics: thermodynamics_manager.ThermodynamicsManager,
      monitor_lib: monitor.Monitor,
      ib: Optional[immersed_boundary_method.ImmersedBoundaryMethod] = None,
      dbg: Optional[components_debug.ComponentsDebug] = None,
  ):
    """Initializes the velocity update library."""
    self._kernel_op = kernel_op
    self._params = params
    self.monitor = monitor_lib

    self.diffusion_fn = diffusion.diffusion_momentum(self._params)

    self._thermodynamics = thermodynamics
    self._halo_dims = (0, 1, 2)
    self._replica_dims = (0, 1, 2)

    self._bc_manager = (
        physical_variable_keys_manager.BoundaryConditionKeysHelper())
    self._src_manager = (physical_variable_keys_manager.SourceKeysHelper())

    self._gravity_vec = (
        self._params.gravity_direction if self._params.gravity_direction else [
            0.0,
        ] * 3)

    self._use_sgs = self._params.use_sgs
    filter_widths = (self._params.dx, self._params.dy, self._params.dz)
    if self._use_sgs:
      self._sgs_model = sgs_model.SgsModel(self._kernel_op, filter_widths,
                                           params.sgs_model)

    self._bc = {
        varname: bc_val
        for varname, bc_val in self._params.bc.items()
        if varname in _KEYS_VELOCITY
    }

    # Define a mapping of shear stress name to directions. In this solver, the
    # shear stress variable is expressed as a dictionary with keys being the
    # directions. The naming of the shear stress in boundary conditions are
    # fully expressed instead of using the two-letter direction indicator to
    # prevent confusion. This map converts the fully expressed shear stress name
    # to directions, which allows the boundary conditions to have the same
    # indices with the shear stress variable.
    self._tau_name_map = {
        'tau00': 'xx',
        'tau01': 'xy',
        'tau02': 'xz',
        'tau10': 'yx',
        'tau11': 'yy',
        'tau12': 'yz',
        'tau20': 'zx',
        'tau21': 'zy',
        'tau22': 'zz',
    }
    self._tau_bc_update_fn = {}
    self._source = {_KEY_U: None, _KEY_V: None, _KEY_W: None}
    self.grad_central = [
        lambda f: self._kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: self._kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: self._kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    ]

    self._ib = ib if ib is not None else (
        immersed_boundary_method.immersed_boundary_method_factory(self._params))

    self._dbg = dbg

  def _exchange_halos(self, f, bc_f, replica_id, replicas):
    """Performs halo exchange for the variable f."""
    return halo_exchange.inplace_halo_exchange(
        f,
        self._halo_dims,
        replica_id,
        replicas,
        self._replica_dims,
        self._params.periodic_dims,
        bc_f,
        width=self._params.halo_width)

  def exchange_velocity_halos(
      self,
      f: FlowFieldVal,
      name: Text,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
  ) -> FlowFieldVal:
    """Performs halo exchange for velocity `f` and updates it's halos.

    Note that the boundary condition will be adjusted prior to the halo
    exchange. For example, values in the ghost cells are updated based on the
    transient fluid field and the boundary if the boundary condition type is
    specified as Dirichlet.

    Args:
      f: The 3D tensor field to which the halo exhange is performed.
      name: The name of the variable. It's used to retrieve the boundary
        condition from the boundary condition library.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.

    Returns:
      A 3D tensor with values in halos updated.
    """
    bc = self._bc
    return self._exchange_halos(f, bc[name], replica_id, replicas)

  def _momentum_update(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      additional_states: FlowFieldMap,
      mu: FlowFieldVal,
      p: FlowFieldVal,
      rho_mix: Optional[FlowFieldVal] = None,
      forces: Sequence[Optional[FlowFieldVal]] = (None, None, None),
      dbg: bool = False,
  ):
    """Provides a function that computes the RHS of the momentum equation.

    This function provides a wrapper for the function that computes the rhs
    `f(rhou)` of the momentum equation in functional form, i.e.
    `d rhou / dt = f(rhou)`.
    Because no boundary constraint is enforced for the shear stress, the
    outermost
    layer, i.e. at `halo_width` = 1, is invalid. And because the momentum
    equation
    use the gradient of the shear stress to compute the diffusion term, cells at
    `halo_width` <= 2 are not valid. Therefore, to use this update function, the
    minimum `halo_width` is 2.
    Note that mu, p, and rho_mix are 3D tensors in form of lists of 2D x-y
    slices.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      additional_states: A dictionary that holds helper variable fields for the
        source term computation.
      mu: Dynamic viscosity of the flow field.
      p: Pressure.
      rho_mix: Density of the inert mixture.
      forces: A three component sequence with each component being the external
        forces applied to its corresponding dimension.
      dbg: An option of whether the debug mode is on. If `True`, a dictionary of
        all terms in the equation will be returned instead of the right hand
        side update.

    Returns:
      momentum_function: A function that computes the `f(rhou)`.
    """

    if rho_mix is None:
      rho_mix = [self._params.rho * tf.ones_like(p_i) for p_i in p]

    zz = additional_states['zz'] if 'zz' in additional_states.keys() else [
        tf.zeros_like(p_i, dtype=p_i.dtype) for p_i in p
    ]
    rho_ref = self._thermodynamics.rho_ref(zz)

    def momentum_function(rho_u: FlowFieldVal, rho_v: FlowFieldVal,
                          rho_w: FlowFieldVal, u: FlowFieldVal, v: FlowFieldVal,
                          w: FlowFieldVal):
      """Computes the functional RHS for the three momentum equations.

      NB: values in `halo_width` <= 2 returned by this function are invalid.

      Args:
        rho_u: Momentum component in the x dimension, with updated boundary
          condition.
        rho_v: Momentum component in the y dimension, with updated boundary
          condition.
        rho_w: Momentum component in the z dimension, with updated boundary
          condition.
        u: Velocity component in the x dimension, with updated boundary
          condition.
        v: Velocity component in the y dimension, with updated boundary
          condition.
        w: Velocity component in the z dimension, with updated boundary
          condition.

      Returns:
        A stack of `FlowFieldVal` of size three, each representing the RHS
        of the momentum equation in the x, y, and z dimension.

      Raises:
        ValueError: If `option` is not one of: 'CENTRAL' 'QUICK'.
      """

      states = {
          _KEY_RHO_U: rho_u,
          _KEY_RHO_V: rho_v,
          _KEY_RHO_W: rho_w,
          _KEY_U: u,
          _KEY_V: v,
          _KEY_W: w,
          _KEY_RHO: rho_mix,
          _KEY_P: p,
      }
      velocity_key = _KEYS_VELOCITY
      momentum_key = _KEYS_MOMENTUM
      h = (self._params.dx, self._params.dy, self._params.dz)
      dt = self._params.dt

      diff_all = self.diffusion_fn(
          self._kernel_op,
          replica_id,
          replicas,
          self._params.diffusion_scheme,
          mu,
          h,
          states,
          tau_bc_update_fn=self._tau_bc_update_fn,
          helper_variables=additional_states)  # pytype: disable=wrong-arg-types

      def momentum_function_in_dim(dim):
        """Computes the RHS of the momentum equation in `dim`."""
        f = states[velocity_key[dim]]

        gravity = eq_utils.buoyancy_source(self._kernel_op, rho_mix, rho_ref,
                                           self._params, dim)

        # Computes the convection term.
        g_corr = None if np.abs(
            self._gravity_vec[dim]) < _G_THRESHOLD else gravity
        if (self._params.convection_scheme !=
            _ConvectionScheme.CONVECTION_SCHEME_WENO):
          # Pressure is used for the Rhie-Chow correction in schemes other than
          # WENO.
          p_corr = [p,] * 3
        else:
          # The pressure gradient term is included in the convection term when
          # the WENO scheme is used. This eliminates the need for the Rhie-Chow
          # correction.
          p_corr = [tf.nest.map_structure(tf.zeros_like, p) for _ in range(3)]
          p_corr[dim] = p
        conv = [
            convection.convection_term(  # pylint: disable=g-complex-comprehension
                self._kernel_op,
                replica_id,
                replicas,
                f,
                states[momentum_key[i]],
                p_corr[i],
                h[i],
                dt,
                i,
                tuple(self._params.bc_type[i]),
                momentum_key[i],
                self._params.halo_width,
                self._params.convection_scheme,
                src=g_corr,
                apply_correction=self._params.enable_rhie_chow_correction)
            for i in range(3)
        ]

        # Computes the diffusion term.
        diff = diff_all[velocity_key[dim]]

        # Computes the pressure gradient.
        if (self._params.convection_scheme ==
            _ConvectionScheme.CONVECTION_SCHEME_WENO):
          # The pressure contribution is considered with the convection when
          # WENO scheme is used to eliminate the need for Rhie-Chow correction.
          dp_dh = tf.nest.map_structure(tf.zeros_like, p)
        elif (self._thermodynamics.solver_mode ==
              thermodynamics_pb2.Thermodynamics.ANELASTIC):
          alpha = tf.nest.map_structure(tf.math.reciprocal, rho_ref)
          a_p = tf.nest.map_structure(tf.math.multiply, alpha, p)
          d_ap_dh = tf.nest.map_structure(lambda d_ap: d_ap / (2.0 * h[dim]),
                                          self.grad_central[dim](a_p))
          # Modified pressure gradient term.
          dp_dh = tf.nest.map_structure(tf.math.multiply, rho_ref, d_ap_dh)
        else:
          dp_dh = tf.nest.map_structure(lambda dp_: dp_ / (2.0 * h[dim]),
                                        self.grad_central[dim](p))

        # Computes external forcing terms.
        force = forces[dim] if forces[dim] else [
            tf.zeros_like(f_i) for f_i in f
        ]

        source_fn = self._params.source_update_fn(velocity_key[dim])

        if source_fn is not None:
          source = source_fn(self._kernel_op, replica_id, replicas, states,
                             additional_states,
                             self._params)[self._src_manager.generate_src_key(
                                 velocity_key[dim])]
          force = tf.nest.map_structure(tf.math.add, force, source)

        if dbg:
          return {
              'conv_x': conv[0],
              'conv_y': conv[1],
              'conv_z': conv[2],
              'diff_x': diff[0],
              'diff_y': diff[1],
              'diff_z': diff[2],
              'gravity': gravity,
              'force': force,
          }

        momentum_terms = zip(conv[0], conv[1], conv[2], diff[0], diff[1],
                             diff[2], dp_dh, gravity, force)

        rhs = [
            -c_x - c_y - c_z + d_x + d_y + d_z - dp_dh_i + gravity_i + force_i
            for c_x, c_y, c_z, d_x, d_y, d_z, dp_dh_i, gravity_i, force_i in
            momentum_terms
        ]

        if self._ib is not None:
          var_name = momentum_key[dim]
          rhs_name = self._ib.ib_rhs_name(var_name)
          helper_states = {
              rhs_name: rhs,
              'ib_interior_mask': additional_states['ib_interior_mask'],
          }
          rhs_ib_updated = self._ib.update_forcing(self._kernel_op, replica_id,
                                                   replicas,
                                                   {var_name: states[var_name]},
                                                   helper_states)
          rhs = rhs_ib_updated[rhs_name]

        return rhs

      rhs_u = momentum_function_in_dim(0)
      rhs_v = momentum_function_in_dim(1)
      rhs_w = momentum_function_in_dim(2)

      return (rhs_u, rhs_v, rhs_w)

    return momentum_function

  def _update_wall_bc(
      self,
      states: FlowFieldMap,
  ) -> None:
    """Updates the boundary conditions for velocity at walls.

    Here we assume that the wall is at the midpoint between the first halo and
    fluid layer. For a velocity component to be 0 at this face, the first halo
    layer that is adjacent to the fluid is set to the opposite of the first
    fluid layer, so that the interpolated value on the face between them is 0.
    Values in the ghost cells are extrapolated linear based on these 2 layres,
    which provides a 0 flux at the face with the QUICK scheme.

    For non-slip walls, all velocity components at the wall are set to zero. So
    treatments described above are applied to 'u', v', and 'w'.

    For free-slip and shear walls, Neumann boundary condition is applied to
    velocity components parallel to the wall so that the velocity gradients for
    these components at the wall is what has been specified.

    Args:
      states: A dictionary that holds flow field variables from the latest
        prediction. Must contain 'u', 'v', and 'w'.
    """

    def bc_planes_for_wall(val, dim, face):
      """Generates a list of planes to be applied as wall boundary condition."""
      bc_planes = []
      for i in range(self._params.halo_width):
        idx = i if face == 1 else self._params.halo_width - 1 - i
        bc_planes.append(
            common_ops.get_face(val, dim, face, self._params.halo_width,
                                -1.0 * (2 * idx + 1))[0])
      return bc_planes

    for dim in range(3):
      for face in range(2):
        if (self._params.bc_type[dim][face]
            not in (boundary_condition_utils.BoundaryType.NON_SLIP_WALL,
                    boundary_condition_utils.BoundaryType.SHEAR_WALL,
                    boundary_condition_utils.BoundaryType.SLIP_WALL)):
          continue

        velocity_components = (
            _KEYS_VELOCITY if self._params.bc_type[dim][face]
            == boundary_condition_utils.BoundaryType.NON_SLIP_WALL else
            [_KEYS_VELOCITY[dim]])

        for velocity_key in velocity_components:
          bc_planes = bc_planes_for_wall(states[velocity_key], dim, face)

          self._bc[velocity_key][dim][face] = (halo_exchange.BCType.DIRICHLET,
                                               bc_planes)

  def update_velocity_halos(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      states_0: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates halos for u, v, and w.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction. Must have 'u', 'v', 'w', 'rho', 'rho_n', and 'rho_0' in it.
      states_0: A dictionary that holds flow field variables from the previous
        time step.

    Returns:
      A dictionary with 'u', 'v' and 'w'. Halos of these fields are updated.
    """
    del states_0

    exchange_halos = functools.partial(
        self.exchange_velocity_halos, replica_id=replica_id, replicas=replicas)

    self._update_wall_bc(states)

    u = exchange_halos(states[_KEY_U], _KEY_U)
    v = exchange_halos(states[_KEY_V], _KEY_V)
    w = exchange_halos(states[_KEY_W], _KEY_W)

    return {_KEY_U: u, _KEY_V: v, _KEY_W: w}

  def _update_helper_variables(
      self,
      states: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) ->...:
    """Generates a dictionary of helper variables for term closures.

    Args:
      states: Prognostic variables in the simulation. Note that all transported
        scalars should be associated with the time step before the temporal
        evolution at the present time step.
      additional_states: State variables that provides additional information
        for model closures that are used in a simulation.

    Returns:
      A dictionary of helper variables specificlaly for the momentum equations.
    """
    helper_variables = {}
    helper_variables.update(additional_states)

    # Add variables required by the Monin-Obukhov similarity theory.
    if (self._params.boundary_models is not None and
        self._params.boundary_models.HasField('most')):

      if 'theta' in states:
        helper_variables.update({'theta': states['theta']})
      elif 'theta_li' in states:
        helper_variables.update({'theta': states['theta_li']})
      elif 'T' in states:
        # Because pressure is the same as the reference pressure on the ground,
        # potential temperatures are equivalent to temperature.
        helper_variables.update({'theta': states['T']})
      elif 'e_t' in states:
        # 'e_t' as a prognostic variable suggests that the water thermodynamics
        # is applied.
        thermal_model = water.Water(self._params)
        q_t = states.get('q_t',
                         tf.nest.map_structure(tf.zeros_like, states['e_t']))
        zz = additional_states.get(
            'zz', tf.nest.map_structure(tf.zeros_like, states['e_t']))
        e = thermal_model.internal_energy_from_total_energy(
            states['e_t'], states[_KEY_U], states[_KEY_V], states[_KEY_W], zz)
        temperature = thermal_model.saturation_adjustment(
            'e_int', e, states[_KEY_RHO], q_t)
        theta = thermal_model.potential_temperatures(temperature, q_t,
                                                     states[_KEY_RHO], zz)
        helper_variables.update({'theta': theta['theta_v']})
      else:
        raise ValueError(
            'One of `theta`, `T`, and `e_t` needs to be provided to use the '
            'Monin-Obukhov similarity theory.')

    return helper_variables

  def prestep(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      additional_states: FlowFieldMap,
  ) -> None:
    """Updates additional information required for velocity step.

    This function is called before the beginning of each time step. It updates
    the boundary conditions of 'u', 'v', 'w', and the shear stresses if
    required. It also updates the forcing term of each momentum components.
    These information will be hold within this helper object.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.
    """
    # Parse additional states to extract boundary conditions.
    self._bc = self._bc_manager.update_helper_variable_from_additional_states(
        additional_states, self._params.halo_width, self._bc)
    for key, val in self._bc.items():
      if key not in self._tau_name_map.keys():
        continue
      self._tau_bc_update_fn.update({
          self._tau_name_map[key]:
              functools.partial(
                  self._exchange_halos,
                  bc_f=val,
                  replica_id=replica_id,
                  replicas=replicas,
              )
      })

    # Parse additional states to extract external source/forcing terms.
    self._source.update(
        self._src_manager.update_helper_variable_from_additional_states(
            additional_states))

  def prediction_step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      states_0: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Predicts the velocity from the momentum equation.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      states_0: A dictionary that holds flow field variables from the previous
        time step.
      additional_states: A dictionary that holds constants that will be used in
        the simulation, e.g. boundary conditions, forcing terms.

    Returns:
      The predicted velocity 'u', 'v', 'w', the momentum 'rho_u', 'rho_v',
      'rho_w', and all debugging terms (if required).
    """
    updated_velocity = {}

    rho_mid = common_ops.average(states[_KEY_RHO], states_0[_KEY_RHO])
    u_mid = common_ops.average(states[_KEY_U], states_0[_KEY_U])
    v_mid = common_ops.average(states[_KEY_V], states_0[_KEY_V])
    w_mid = common_ops.average(states[_KEY_W], states_0[_KEY_W])

    states_mid = {key: val for key, val in states_0.items()}
    states_mid.update({_KEY_U: u_mid, _KEY_V: v_mid, _KEY_W: w_mid})
    helper_variables = self._update_helper_variables(states_mid,
                                                     additional_states)

    if self._use_sgs:
      nu_t = self._sgs_model.turbulent_viscosity(
          (u_mid, v_mid, w_mid),
          replicas=replicas,
          additional_states=additional_states)
      mu = [(self._params.nu + nu_t_i) * rho_i
            for nu_t_i, rho_i in zip(nu_t, states_0[_KEY_RHO])]
    else:
      mu = [self._params.nu * rho_i for rho_i in states_0[_KEY_RHO]]

    forces = [self._source[_KEY_U], self._source[_KEY_V], self._source[_KEY_W]]

    momentum_rhs = self._momentum_update(replica_id, replicas, helper_variables,
                                         mu, states[_KEY_P],
                                         states_0['rho_thermal'], forces)

    rho_u, rho_v, rho_w = time_integration.time_advancement_explicit(
        functools.partial(momentum_rhs, u=u_mid, v=v_mid, w=w_mid),
        self._params.dt, self._params.time_integration_scheme,
        (states_0[_KEY_RHO_U], states_0[_KEY_RHO_V], states_0[_KEY_RHO_W]),
        (states[_KEY_RHO_U], states[_KEY_RHO_V], states[_KEY_RHO_W]))

    if self._dbg is not None:
      rho_u_m = common_ops.average(states_0[_KEY_RHO_U], rho_u)
      rho_v_m = common_ops.average(states_0[_KEY_RHO_V], rho_v)
      rho_w_m = common_ops.average(states_0[_KEY_RHO_W], rho_w)
      terms = self._momentum_update(replica_id, replicas, additional_states, mu,
                                    states[_KEY_P], states_0['rho_thermal'],
                                    forces, True)(rho_u_m, rho_v_m, rho_w_m,
                                                  u_mid, v_mid, w_mid)
      updated_velocity.update(self._dbg.update_momentum_terms(terms))

    states_buf = {
        _KEY_U: tf.nest.map_structure(tf.math.divide, rho_u, rho_mid),
        _KEY_V: tf.nest.map_structure(tf.math.divide, rho_v, rho_mid),
        _KEY_W: tf.nest.map_structure(tf.math.divide, rho_w, rho_mid),
        _KEY_RHO: states[_KEY_RHO],
    }
    states_updated = self.update_velocity_halos(replica_id, replicas,
                                                states_buf, states_0)
    u = states_updated[_KEY_U]
    v = states_updated[_KEY_V]
    w = states_updated[_KEY_W]

    # Applies the marker-and-cell or Cartesian grid method if requested in the
    # config file.
    if self._ib is not None:
      velocity_ib_updated = self._ib.update_states(self._kernel_op, replica_id,
                                                   replicas, {
                                                       _KEY_U: u,
                                                       _KEY_V: v,
                                                       _KEY_W: w
                                                   }, additional_states,
                                                   self._bc)
      u = velocity_ib_updated[_KEY_U]
      v = velocity_ib_updated[_KEY_V]
      w = velocity_ib_updated[_KEY_W]

    updated_velocity.update({
        _KEY_U: u,
        _KEY_V: v,
        _KEY_W: w,
        _KEY_RHO_U: tf.nest.map_structure(tf.math.multiply, rho_mid, u),
        _KEY_RHO_V: tf.nest.map_structure(tf.math.multiply, rho_mid, v),
        _KEY_RHO_W: tf.nest.map_structure(tf.math.multiply, rho_mid, w),
    })

    if 'nu_t' in additional_states.keys() and self._use_sgs:
      updated_velocity.update({'nu_t': nu_t})

    return updated_velocity  # pytype: disable=bad-return-type

  def correction_step(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      states: FlowFieldMap,
      states_0: FlowFieldMap,
      additional_states: FlowFieldMap,
  ) -> FlowFieldMap:
    """Updates the momentum and velocity from the pressure correction.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      states: A dictionary that holds flow field variables from the latest
        prediction.
      states_0: A dictionary that holds flow field variables from the previous
        time step.
      additional_states: A dictionary that holds constants that will be used in
        the simulation. Must include 'dp'.

    Returns:
      The updated velocity and momentum with pressure correction.
    """
    dt = self._params.dt

    rho_mid = common_ops.average(states[_KEY_RHO], states_0[_KEY_RHO])
    if (self._thermodynamics.solver_mode ==
        thermodynamics_pb2.Thermodynamics.ANELASTIC):
      dp = tf.nest.map_structure(tf.math.divide, states[common.KEY_DP], rho_mid)
    else:
      dp = states[common.KEY_DP]

    def generate_correction_fn(dim):
      dh = (self._params.dx, self._params.dy, self._params.dz)[dim]

      def correction_fn(rho_u_, rho_, dp_):
        if (self._thermodynamics.solver_mode ==
            thermodynamics_pb2.Thermodynamics.ANELASTIC):
          return rho_u_ - dt * rho_ * dp_ / (2 * dh)
        else:
          return rho_u_ - dt * dp_ / (2 * dh)

      return correction_fn

    velocity_keys = (_KEY_RHO_U, _KEY_RHO_V, _KEY_RHO_W)
    states_new = {
        velocity_keys[dim]: tf.nest.map_structure(
            generate_correction_fn(dim), states[velocity_keys[dim]], rho_mid,
            self.grad_central[dim](dp)) for dim in range(3)
    }

    states_buf = {
        _KEY_U:
            tf.nest.map_structure(tf.math.divide, states_new[_KEY_RHO_U],
                                  rho_mid),
        _KEY_V:
            tf.nest.map_structure(tf.math.divide, states_new[_KEY_RHO_V],
                                  rho_mid),
        _KEY_W:
            tf.nest.map_structure(tf.math.divide, states_new[_KEY_RHO_W],
                                  rho_mid),
        _KEY_RHO:
            states[_KEY_RHO],
    }

    # Applies the marker-and-cell or Cartesian grid method if requested in the
    # config file.
    if self._ib is not None:
      states_buf = self._ib.update_states(self._kernel_op, replica_id, replicas,
                                          states_buf, additional_states,
                                          self._bc)

    states_new.update(
        self.update_velocity_halos(replica_id, replicas, states_buf, states_0))

    # Updates monitors.
    monitor_vars = {}
    # Exports momentum with B.C. and halos updated.
    for uvw in _KEYS_VELOCITY:
      monitor_key = monitor.MONITOR_KEY_TEMPLATE.format(
          module='velocity',
          statistic_type='raw',
          metric_name='rho-{}'.format(uvw))
      if self.monitor.check_key(monitor_key):
        monitor_vars.update({
            monitor_key:
                tf.stack(
                    tf.nest.map_structure(tf.math.multiply, states_new[uvw],
                                          rho_mid)),
        })

    states_new.update(monitor_vars)

    return states_new
