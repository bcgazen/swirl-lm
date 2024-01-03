# Copyright 2023 The swirl_lm Authors.
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

"""Utility functions that are commonly used in different equations."""

from typing import Callable, Dict, Optional, Text
import numpy as np
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.boundary_condition import monin_obukhov_similarity_theory
from swirl_lm.numerics import calculus
from swirl_lm.numerics import filters
from swirl_lm.physics import constants
from swirl_lm.physics.thermodynamics import thermodynamics_pb2
from swirl_lm.utility import common_ops
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import types
import tensorflow as tf

FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap

# Parameters required by source terms due to subsidence velocity. Reference:
# Siebesma, A. Pier, A. Pier Siebesma, Christopher S. Bretherton, Andrew Brown,
# Andreas Chlond, Joan Cuxart, Peter G. Duynkerke, et al. 2003. “A Large Eddy
# Simulation Intercomparison Study of Shallow Cumulus Convection.” Journal of
# the Atmospheric Sciences.
_W_MAX = -0.65e-2
_Z_F1 = 1500.0
_Z_F5 = 2100.0
# Parameter required by the large-scale subsidence velocity, units 1/s.
# Reference:
# Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman, Christopher S. Bretherton,
# Andreas Chlond, Stephan de Roode, James Edwards, et al. 2005. “Evaluation of
# Large-Eddy Simulations via Observations of Nocturnal Marine Stratocumulus.”
# Monthly Weather Review 133 (6): 1443–62.
_D = 3.75e-6


def shear_stress(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    mu: FlowFieldVal,
    dx: float,
    dy: float,
    dz: float,
    u: FlowFieldVal,
    v: FlowFieldVal,
    w: FlowFieldVal,
    shear_bc_update_fn: Optional[Dict[Text, Callable[[FlowFieldVal],
                                                     FlowFieldVal]]] = None,
) -> FlowFieldMap:
  """Computes the viscous shear stress.

  The shear stress is computed as:
    τᵢⱼ = μ [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ], i ≠ j
    τᵢⱼ = 2 μ [∂uᵢ/∂xⱼ - 1/3 ∂uₖ/∂xₖ δᵢⱼ], i = j
  Note that u, v, w are 3D tensors that are represented in the form of a list of
  2D x-y slices.

  Args:
    kernel_op: An object holding a library of kernel operations.
    mu: Dynamic viscosity of the flow field.
    dx: Grid spacing in the x dimension.
    dy: Grid spacing in the y dimension.
    dz: Grid spacing in the z dimension.
    u: Velocity component in the x dimension, with updated boundary condition.
    v: Velocity component in the y dimension, with updated boundary condition.
    w: Velocity component in the z dimension, with updated boundary condition.
    shear_bc_update_fn: A dictionary of halo_exchange functions for the shear
      stress tensor.

  Returns:
    The 9 component stress stress tensor for each grid point. Values in the halo
    with width 1 is invalid.
  """
  du_dx = calculus.grad(kernel_op, [u, v, w], [dx, dy, dz])

  du_11 = du_dx[0][0]
  du_12 = du_dx[0][1]
  du_13 = du_dx[0][2]
  du_21 = du_dx[1][0]
  du_22 = du_dx[1][1]
  du_23 = du_dx[1][2]
  du_31 = du_dx[2][0]
  du_32 = du_dx[2][1]
  du_33 = du_dx[2][2]

  s11 = du_11
  s12 = tf.nest.map_structure(common_ops.average, du_12, du_21)
  s13 = tf.nest.map_structure(common_ops.average, du_13, du_31)
  s21 = s12
  s22 = du_22
  s23 = tf.nest.map_structure(common_ops.average, du_23, du_32)
  s31 = s13
  s32 = s23
  s33 = du_33

  du_kk = tf.nest.map_structure(lambda x, y, z: x + y + z, du_11, du_22, du_33)

  tau_ij = lambda mu, s_ij: 2 * mu * s_ij
  tau_ii = lambda mu, s_ii, div_u: 2 * mu * (s_ii - div_u / 3)

  tau11 = tf.nest.map_structure(tau_ii, mu, s11, du_kk)
  tau12 = tf.nest.map_structure(tau_ij, mu, s12)
  tau13 = tf.nest.map_structure(tau_ij, mu, s13)
  tau21 = tf.nest.map_structure(tau_ij, mu, s21)
  tau22 = tf.nest.map_structure(tau_ii, mu, s22, du_kk)
  tau23 = tf.nest.map_structure(tau_ij, mu, s23)
  tau31 = tf.nest.map_structure(tau_ij, mu, s31)
  tau32 = tf.nest.map_structure(tau_ij, mu, s32)
  tau33 = tf.nest.map_structure(tau_ii, mu, s33, du_kk)

  tau = {
      'xx': tau11,
      'xy': tau12,
      'xz': tau13,
      'yx': tau21,
      'yy': tau22,
      'yz': tau23,
      'zx': tau31,
      'zy': tau32,
      'zz': tau33,
  }

  if shear_bc_update_fn:
    for key, fn in shear_bc_update_fn.items():
      tau.update({key: fn(tau[key])})

  return tau


def shear_flux(params: parameters_lib.SwirlLMParameters) -> ...:
  """Generates a function that computes the shear fluxes at cell faces.

  Args:
    params: A object of the simulation parameter context. `boundary_models.most`
      is used here.

  Returns:
    A function that computes the 9 component shear stress tensor.
  """
  if (params.boundary_models is not None and
      params.boundary_models.HasField('most')):
    most = (
        monin_obukhov_similarity_theory.monin_obukhov_similarity_theory_factory(
            params))
  else:
    most = None

  def shear_flux_fn(
      kernel_op: get_kernel_fn.ApplyKernelOp,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      mu: FlowFieldVal,
      dx: float,
      dy: float,
      dz: float,
      u: FlowFieldVal,
      v: FlowFieldVal,
      w: FlowFieldVal,
      helper_variables: Optional[FlowFieldMap] = None,
  ) -> FlowFieldMap:
    """Computes the viscous shear stress on the cell faces.

    The shear stress is computed as:
      τᵢⱼ = μ [∂uᵢ/∂xⱼ + ∂uⱼ/∂xᵢ], i ≠ j
      τᵢⱼ = 2 μ [∂uᵢ/∂xⱼ - 1/3 ∂uₖ/∂xₖ δᵢⱼ], i = j
    Note that u, v, w are 3D tensors that are represented in the form of a list
    of 2D x-y slices.

    Locations of the fluxes:
      tau11/tau_xx: x face, i - 1/2 stored at i;
      tau12/tau_xy: y face, j - 1/2 stored at j;
      tau13/tau_xz: z face, k - 1/2 stored at k;
      tau21/tau_yx: x face, i - 1/2 stored at i;
      tau22/tau_yy: y face, j - 1/2 stored at j;
      tau23/tau_yz: z face, k - 1/2 stored at k;
      tau31/tau_zx: x face, i - 1/2 stored at i;
      tau32/tau_zy: y face, j - 1/2 stored at j;
      tau33/tau_zz: z face, k - 1/2 stored at k.

    Args:
      kernel_op: An object holding a library of kernel operations.
      replica_id: The index of the current TPU replica.
      replicas: A numpy array that maps grid coordinates to replica id numbers.
      mu: Dynamic viscosity of the flow field.
      dx: Grid spacing in the x dimension.
      dy: Grid spacing in the y dimension.
      dz: Grid spacing in the z dimension.
      u: Velocity component in the x dimension, with updated boundary condition.
      v: Velocity component in the y dimension, with updated boundary condition.
      w: Velocity component in the z dimension, with updated boundary condition.
      helper_variables: A dictionarry that stores variables that provides
        additional information for computing the diffusion term, e.g. the
        potential temperature for the Monin-Obukhov similarity theory.

    Returns:
      The 9 component stress tensor for each grid point. Values in the halo with
      width 1 are invalid.
    """

    def interp(f: FlowFieldVal, dim: int) -> FlowFieldVal:
      """Interpolates `value` in `dim` onto faces (i - 1/2 stored at i)."""
      if dim == 0:
        df = kernel_op.apply_kernel_op_x(f, 'ksx')
      elif dim == 1:
        df = kernel_op.apply_kernel_op_y(f, 'ksy')
      elif dim == 2:
        df = kernel_op.apply_kernel_op_z(f, 'ksz', 'kszsh')
      else:
        raise ValueError('Unsupport dimension: {}'.format(dim))

      return tf.nest.map_structure(lambda df_i: 0.5 * df_i, df)

    def grad_n(f: FlowFieldVal, dim: int, h: float) -> FlowFieldVal:
      """Computes gradient of `value` in `dim` on nodes."""
      if dim == 0:
        df = kernel_op.apply_kernel_op_x(f, 'kDx')
      elif dim == 1:
        df = kernel_op.apply_kernel_op_y(f, 'kDy')
      elif dim == 2:
        df = kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh')
      else:
        raise ValueError('Unsupport dimension: {}'.format(dim))

      return tf.nest.map_structure(lambda df_i: df_i / (2.0 * h), df)

    def grad_f(f: FlowFieldVal, dim: int, h: float) -> FlowFieldVal:
      """Computes gradient of `value` in `dim` on faces."""
      if dim == 0:
        df = kernel_op.apply_kernel_op_x(f, 'kdx')
      elif dim == 1:
        df = kernel_op.apply_kernel_op_y(f, 'kdy')
      elif dim == 2:
        df = kernel_op.apply_kernel_op_z(f, 'kdz', 'kdzsh')
      else:
        raise ValueError('Unsupport dimension: {}'.format(dim))

      return tf.nest.map_structure(lambda df_i: df_i / h, df)

    def grad_interp(f: FlowFieldVal, grad_dim: int, interp_dim: int,
                    h: float) -> FlowFieldVal:
      """Computes gradient of `value` in `grad_dim` on faces in `interp_dim`."""
      return interp(grad_n(f, grad_dim, h), interp_dim)

    s11 = grad_f(u, 0, dx)
    # Note that `du/dy` is on j faces, but `dv/dx` is on i faces if we compute
    # it directly based on `v` at nodes. Therefore, to get S12 on j faces,
    # `dv/dx` (computed with a central-difference scheme) needs to be
    # interpolated onto j faces. Similar interpolations are applied to compute
    # other components in the strain rate tensor S.
    s12 = tf.nest.map_structure(
        lambda du_dy, dv_dx: 0.5 * (du_dy + dv_dx),
        grad_f(u, 1, dy),
        grad_interp(v, 0, 1, dx),
    )
    s13 = tf.nest.map_structure(
        lambda du_dz, dw_dx: 0.5 * (du_dz + dw_dx),
        grad_f(u, 2, dz),
        grad_interp(w, 0, 2, dx),
    )
    s21 = tf.nest.map_structure(
        lambda dv_dx, du_dy: 0.5 * (dv_dx + du_dy),
        grad_f(v, 0, dx),
        grad_interp(u, 1, 0, dy),
    )
    s22 = grad_f(v, 1, dy)
    s23 = tf.nest.map_structure(
        lambda dv_dz, dw_dy: 0.5 * (dv_dz + dw_dy),
        grad_f(v, 2, dz),
        grad_interp(w, 1, 2, dy),
    )
    s31 = tf.nest.map_structure(
        lambda dw_dx, du_dz: 0.5 * (dw_dx + du_dz),
        grad_f(w, 0, dx),
        grad_interp(u, 2, 0, dz),
    )
    s32 = tf.nest.map_structure(
        lambda dw_dy, dv_dz: 0.5 * (dw_dy + dv_dz),
        grad_f(w, 1, dy),
        grad_interp(v, 2, 1, dz),
    )
    s33 = grad_f(w, 2, dz)

    du_kk_x = tf.nest.map_structure(
        lambda du_dx, dv_dy, dw_dz: du_dx + dv_dy + dw_dz,
        s11,
        grad_interp(v, 1, 0, dy),
        grad_interp(w, 2, 0, dz),
    )

    du_kk_y = tf.nest.map_structure(
        lambda du_dx, dv_dy, dw_dz: du_dx + dv_dy + dw_dz,
        grad_interp(u, 0, 1, dx),
        s22,
        grad_interp(w, 2, 1, dz),
    )

    du_kk_z = tf.nest.map_structure(
        lambda du_dx, dv_dy, dw_dz: du_dx + dv_dy + dw_dz,
        grad_interp(u, 0, 2, dx),
        grad_interp(v, 1, 2, dy),
        s33,
    )

    tau11 = tf.nest.map_structure(
        lambda mu_i, s11_i, du_kk_i: 2.0 * mu_i * (s11_i - 1.0 / 3.0 * du_kk_i),
        interp(mu, 0),
        s11,
        du_kk_x,
    )
    tau12 = tf.nest.map_structure(
        lambda mu_i, s12_i: 2.0 * mu_i * s12_i, interp(mu, 1), s12
    )
    tau13 = tf.nest.map_structure(
        lambda mu_i, s13_i: 2.0 * mu_i * s13_i, interp(mu, 2), s13
    )
    tau21 = tf.nest.map_structure(
        lambda mu_i, s21_i: 2.0 * mu_i * s21_i, interp(mu, 0), s21
    )
    tau22 = tf.nest.map_structure(
        lambda mu_i, s22_i, du_kk_i: 2.0 * mu_i * (s22_i - 1.0 / 3.0 * du_kk_i),
        interp(mu, 1),
        s22,
        du_kk_y,
    )
    tau23 = tf.nest.map_structure(
        lambda mu_i, s23_i: 2.0 * mu_i * s23_i, interp(mu, 2), s23
    )
    tau31 = tf.nest.map_structure(
        lambda mu_i, s31_i: 2.0 * mu_i * s31_i, interp(mu, 0), s31
    )
    tau32 = tf.nest.map_structure(
        lambda mu_i, s32_i: 2.0 * mu_i * s32_i, interp(mu, 1), s32
    )
    tau33 = tf.nest.map_structure(
        lambda mu_i, s33_i, du_kk_i: 2.0 * mu_i * (s33_i - 1.0 / 3.0 * du_kk_i),
        interp(mu, 2),
        s33,
        du_kk_z,
    )

    # Add the closure from Monin-Obukhov similarity theory if requested.
    if most is not None:
      if 'theta' not in helper_variables:
        raise ValueError('`theta` is missing for the MOS model.')

      helper_vars = {'u': u, 'v': v, 'w': w, 'theta': helper_variables['theta']}

      # Get the surface shear stress.
      tau_s1, tau_s2, _ = most.surface_shear_stress_and_heat_flux_update_fn(
          helper_vars)

      # The sign of the shear stresses need to be reversed to be consistent with
      # the diffusion scheme.
      tau_s1 = tf.nest.map_structure(lambda t: -t, tau_s1)
      tau_s2 = tf.nest.map_structure(lambda t: -t, tau_s2)

      if most.vertical_dim == 2:
        tau_s1 = [tau_s1]
        tau_s2 = [tau_s2]

      # Replace the shear stress at the ground surface with the MOS closure.
      core_index = 0
      plane_index = params.halo_width
      if most.vertical_dim == 0:
        # `tau_s1` corresponds to the first shear stress component for the v
        # velocity, and `tau_s2` corresponds to the first shear stress component
        # for the w velocity.
        tau21 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau21, most.vertical_dim, core_index,
            plane_index, tau_s1)
        tau31 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau31, most.vertical_dim, core_index,
            plane_index, tau_s2)
      elif most.vertical_dim == 1:
        # `tau_s1` corresponds to the second shear stress component for the u
        # velocity, and `tau_s2` corresponds to the second shear stress
        # component for the w velocity.
        tau12 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau12, most.vertical_dim, core_index,
            plane_index, tau_s1)
        tau32 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau32, most.vertical_dim, core_index,
            plane_index, tau_s2)
      elif most.vertical_dim == 2:
        # `tau_s1` corresponds to the third shear stress component for the u
        # velocity, and `tau_s2` corresponds to the third shear stress component
        # for the v velocity.
        tau13 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau13, most.vertical_dim, core_index,
            plane_index, tau_s1)
        tau23 = common_ops.tensor_scatter_1d_update_global(
            replica_id, replicas, tau23, most.vertical_dim, core_index,
            plane_index, tau_s2)
      else:
        raise ValueError('Unsupport dimension: {}'.format(most.vertical_dim))

    return {
        'xx': tau11,
        'xy': tau12,
        'xz': tau13,
        'yx': tau21,
        'yy': tau22,
        'yz': tau23,
        'zx': tau31,
        'zy': tau32,
        'zz': tau33,
    }

  return shear_flux_fn


def subsidence_velocity_stevens(zz: FlowFieldVal) -> FlowFieldVal:
  """Computes the subsidence velocity following the Stevens' [1] formulation.

  Reference:
  1. Stevens, Bjorn, Chin-Hoh Moeng, Andrew S. Ackerman,
     Christopher S. Bretherton, Andreas Chlond, Stephan de Roode, James Edwards,
     et al. 2005. “Evaluation of Large-Eddy Simulations via Observations of
     Nocturnal Marine Stratocumulus.” Monthly Weather Review 133 (6): 1443–62.

  Args:
    zz: The coordinates in the vertical direction.

  Returns:
    The subsidence velocity.
  """
  return tf.nest.map_structure(lambda z: -_D * z, zz)


def subsidence_velocity_siebesma(zz: FlowFieldVal) -> FlowFieldVal:
  """Computes the subsidence velocity following the Siebesma's [1] formulation.

  Reference:
  1. Siebesma, A. Pier, A. Pier Siebesma, Christopher S. Bretherton,
  Andrew Brown, Andreas Chlond, Joan Cuxart, Peter G. Duynkerke, et al. 2003.
  “A Large Eddy Simulation Intercomparison Study of Shallow Cumulus
  Convection.” Journal of the Atmospheric Sciences.

  Args:
    zz: The coordinates in the vertical direction.

  Returns:
    The subsidence velocity.
  """
  w = [
      tf.compat.v1.where(
          tf.less_equal(z, _Z_F1), _W_MAX * z / _Z_F1,
          _W_MAX * (1.0 - (z - _Z_F1) / (_Z_F5 - _Z_F1))) for z in zz
  ]
  return [
      tf.compat.v1.where(tf.less_equal(z, _Z_F5), w_i, tf.zeros_like(w_i))
      for z, w_i in zip(zz, w)
  ]


def source_by_subsidence_velocity(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    rho: FlowFieldVal,
    height: FlowFieldVal,
    h: float,
    field: FlowFieldVal,
    vertical_dim: int,
) -> FlowFieldVal:
  """Computes the source term for `field` due to subsidence velocity.

  Args:
    kernel_op: A library of finite difference operators.
    rho: The density of the flow field.
    height: The coordinates in the direction vertical to the ground.
    h: The grid spacing in the vertical direction discretization.
    field: The quantity to which the source term is computed.
    vertical_dim: The vertical dimension that is aligned with gravity.

  Returns:
    The source term for `field` due to the subsidence velocity.
  """
  if vertical_dim == 0:
    df = kernel_op.apply_kernel_op_x(field, 'kDx')
  elif vertical_dim == 1:
    df = kernel_op.apply_kernel_op_y(field, 'kDy')
  else:  # vertical_dim == 2
    df = kernel_op.apply_kernel_op_z(field, 'kDz', 'kDzsh')

  df_dh = tf.nest.map_structure(lambda df_i: df_i / (2.0 * h), df)
  w = subsidence_velocity_stevens(height)
  return tf.nest.map_structure(
      lambda rho_i, w_i, df_dh_i: -rho_i * w_i * df_dh_i, rho, w, df_dh
  )


def buoyancy_source(
    kernel_op: get_kernel_fn.ApplyKernelOp,
    rho: FlowFieldVal,
    rho_0: FlowFieldVal,
    params: parameters_lib.SwirlLMParameters,
    dim: int,
) -> FlowFieldVal:
  """Computes the gravitational force of the momentum equation.

  Args:
    kernel_op: A library of finite difference operators.
    rho: The density of the flow field.
    rho_0: The reference density of the environment.
    params: The simulation parameter context. `thermodynamics.solver_mode` is
      used here.
    dim: The spatial dimension that this source corresponds to.

  Returns:
    The source term of the momentum equation due to buoyancy.
  """
  def drho_fn(rho_i, rho_0_i):
    if params.solver_mode == thermodynamics_pb2.Thermodynamics.ANELASTIC:
      return (rho_i - rho_0_i) * rho_0_i / rho_i
    else:
      return rho_i - rho_0_i

  # Computes the gravitational force.
  drho = filters.filter_op(
      kernel_op,
      tf.nest.map_structure(drho_fn, rho, rho_0),
      order=2)
  return tf.nest.map_structure(
      lambda drho_i: drho_i * params.gravity_direction[dim] * constants.G, drho)
