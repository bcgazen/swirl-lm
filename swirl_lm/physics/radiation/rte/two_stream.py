# Copyright 2024 The swirl_lm Authors.
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

# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A library for solving the two-stream radiative transfer equation."""

from typing import Dict, Optional

import numpy as np
from swirl_lm.physics import constants
from swirl_lm.physics.radiation.config import radiative_transfer_pb2
from swirl_lm.physics.radiation.optics import atmospheric_state
from swirl_lm.physics.radiation.optics import optics
from swirl_lm.physics.radiation.rte import monochromatic_two_stream
import swirl_lm.physics.radiation.rte.rte_utils as utils
from swirl_lm.utility import get_kernel_fn
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


AtmosphericState = atmospheric_state.AtmosphericState
FlowFieldMap = types.FlowFieldMap
FlowFieldVal = types.FlowFieldVal


class TwoStreamSolver:
  """A library for solving the two-stream radiative transfer equation.

  Attributes:
    atmospheric_state: An instance of `AtmosphericState` containing volume
      mixing ratio profiles for prevalent atmospheric gases and flux boundary
      conditions at particular times and geographic locations.
  """

  def __init__(
      self,
      radiation_params: radiative_transfer_pb2.RadiativeTransfer,
      grid_params: grid_parametrization.GridParametrization,
      kernel_op: get_kernel_fn.ApplyKernelOp,
      g_dim: int,
  ):
    self.atmospheric_state = AtmosphericState.from_proto(
        radiation_params.atmospheric_state
    )
    self._g_dim = g_dim
    self._halos = grid_params.halo_width
    self._optics_lib = optics.optics_factory(
        radiation_params.optics,
        kernel_op,
        g_dim,
        self._halos,
        self.atmospheric_state.vmr,
    )
    self._monochrom_solver = (
        monochromatic_two_stream.MonochromaticTwoStreamSolver(
            grid_params, kernel_op, g_dim
        )
    )
    self._rte_utils = utils.RTEUtils(grid_params)

    # Operators used when computing heating rate from fluxes.
    self._grad_central = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kDx'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kDy'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kDz', 'kDzsh'),
    )[g_dim]
    self._grad_forward_fn = (
        lambda f: kernel_op.apply_kernel_op_x(f, 'kdx+'),
        lambda f: kernel_op.apply_kernel_op_y(f, 'kdy+'),
        lambda f: kernel_op.apply_kernel_op_z(f, 'kdz+', 'kdz+sh'),
    )[g_dim]

    # Longwave parameters.
    self._top_flux_down_lw = self.atmospheric_state.toa_flux_lw
    self._sfc_emissivity_lw = self.atmospheric_state.sfc_emis

    # Shortwave parameters.
    self._sfc_albedo = self.atmospheric_state.sfc_alb
    self._zenith = self.atmospheric_state.zenith
    self._total_solar_irrad = self.atmospheric_state.irrad
    self._solar_fraction_by_gpt = self._optics_lib.solar_fraction_by_gpt

  def solve_lw(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      vmr_fields: Optional[Dict[str, FlowFieldVal]] = None,
      sfc_temperature: Optional[FlowFieldVal] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Solves two-stream radiative transfer equation over the longwave spectrum.

    Local optical properties like optical depth, single-scattering albedo, and
    asymmetry factor are computed using an optics library and transformed to
    two-stream approximations of reflectance and transmittance. The sources of
    longwave radiation are the Planck sources, which are a function only of
    temperature. To obtain the cell-centered directional Planck sources, the
    sources are first computed at the cell boundaries and the net source
    emanating from the grid cell is determined. Each spectral interval,
    represented by a g-point, is a separate radiative transfer problem, and can
    be computed in parallel. Finally, the independently solved fluxes are summed
    over the full spectrum to yield the final upwelling and downwelling fluxes.
    A `tf.while_loop` is used to loop over the spectrum and the parallelism is
    set to the exact number of g-points to be processed, which ensures that all
    g-points are processed in parallel. If the problem size exceeds the memory
    capacity, the `parallel_iterations` argument can be adjusted accordingly.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      pressure: The pressure field [Pa].
      temperature: The temperature field [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by the chemical formula.
      sfc_temperature: The optional surface temperature represented as either a
        3D `tf.Tensor` or as a list of 2D `tf.Tensor`s but having a single
        vertical dimension [K].
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].

    Returns:
      A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling longwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling longwave radiative flux at face i - 1/2.
      `flux_net`: The net longwave radiative flux at face i - 1/2.
    """
    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    if vmr_fields is not None and self._optics_lib.gas_optics_lw is not None:
      vmr_fields = {
          self._optics_lib.gas_optics_lw.idx_gases[k]: v
          for k, v in vmr_fields.items()
      }

    def step_fn(igpt, cumulative_flux):
      lw_optical_props = dict(
          self._optics_lib.compute_lw_optical_properties(
              pressure,
              temperature,
              molecules,
              igpt,
              vmr_fields=vmr_fields,
              cloud_r_eff_liq=cloud_r_eff_liq,
              cloud_path_liq=cloud_path_liq,
              cloud_r_eff_ice=cloud_r_eff_ice,
              cloud_path_ice=cloud_path_ice,
          )
      )
      planck_srcs = dict(
          self._optics_lib.compute_planck_sources(
              replica_id,
              replicas,
              pressure,
              temperature,
              igpt,
              vmr_fields,
              sfc_temperature=sfc_temperature,
          )
      )
      combined_srcs = self._monochrom_solver.lw_combine_sources(planck_srcs)
      lw_optical_props['level_src_bottom'] = combined_srcs['planck_src_bottom']
      lw_optical_props['level_src_top'] = combined_srcs['planck_src_top']
      optical_props_2stream = (
          self._monochrom_solver.lw_cell_source_and_properties(
              **lw_optical_props,
          )
      )
      # Boundary conditions.
      sfc_src = planck_srcs.get('planck_src_sfc', self._rte_utils.slice(
          planck_srcs['planck_src_bottom'],
          self._g_dim,
          face=0,
          idx=self._halos,
      ))
      top_flux_down = tf.nest.map_structure(
          lambda x: self._top_flux_down_lw * tf.ones_like(x), sfc_src
      )
      sfc_emissivity = tf.nest.map_structure(
          lambda x: self._sfc_emissivity_lw * tf.ones_like(x), sfc_src
      )
      fluxes = self._monochrom_solver.lw_transport(
          replica_id,
          replicas,
          sfc_src=sfc_src,
          top_flux_down=top_flux_down,
          sfc_emissivity=sfc_emissivity,
          **optical_props_2stream,
      )
      return igpt + 1, tf.nest.map_structure(
          tf.math.add, fluxes, cumulative_flux,
      )

    stop_condition = lambda i, states: i < self._optics_lib.n_gpt_lw
    lw_fluxes0 = {
        k: tf.nest.map_structure(tf.zeros_like, pressure)
        for k in ['flux_up', 'flux_down', 'flux_net']
    }
    i0 = tf.constant(0)
    _, fluxes = tf.nest.map_structure(
        tf.stop_gradient,
        tf.while_loop(
            cond=stop_condition,
            body=step_fn,
            loop_vars=(i0, lw_fluxes0),
            parallel_iterations=self._optics_lib.n_gpt_lw,
        ),
    )
    return fluxes

  def solve_sw(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      pressure: FlowFieldVal,
      temperature: FlowFieldVal,
      molecules: FlowFieldVal,
      vmr_fields: Optional[Dict[str, FlowFieldVal]] = None,
      cloud_r_eff_liq: Optional[FlowFieldVal] = None,
      cloud_path_liq: Optional[FlowFieldVal] = None,
      cloud_r_eff_ice: Optional[FlowFieldVal] = None,
      cloud_path_ice: Optional[FlowFieldVal] = None,
  ) -> FlowFieldMap:
    """Solves the two-stream radiative transfer equation for shortwave.

    Local optical properties like optical depth, single-scattering albedo, and
    asymmetry factor are computed using an optics library and transformed to
    two-stream approximations of reflectance and transmittance. The sources of
    shortwave radiation are determined by the diffuse propagation of direct
    solar radiation through the layered atmosphere. Each spectral interval,
    represented by a g-point, is a separate radiative transfer problem, and can
    be computed in parallel. Finally, the independently solved fluxes are summed
    over the full spectrum to yield the final upwelling and downwelling fluxes.
    A `tf.while_loop` is used to loop over the spectrum and the parallelism is
    set to the exact number of g-points to be processed, which ensures that all
    g-points are processed in parallel. If the problem size exceeds the memory
    capacity, the `parallel_iterations` argument can be adjusted accordingly.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      pressure: The pressure field [Pa].
      temperature: The temperature field [K].
      molecules: The number of molecules in an atmospheric grid cell per area
        [molecules/m²].
      vmr_fields: An optional dictionary containing precomputed volume mixing
        ratio fields, keyed by gas index.
      cloud_r_eff_liq: The effective radius of cloud droplets [m].
      cloud_path_liq: The cloud liquid water path in each atmospheric grid cell
        [kg/m²].
      cloud_r_eff_ice: The effective radius of cloud ice particles [m].
      cloud_path_ice: The cloud ice water path in each atmospheric grid cell
        [kg/m²].

    Returns:
      A dictionary with the following entries (in units of W/m²):
      `flux_up`: The upwelling shortwave radiative flux at cell face i - 1/2.
      `flux_down`: The downwelling shortwave radiative flux at face i - 1/2.
      `flux_net`: The net shortwave radiative flux at face i - 1/2.
    """
    def field_like(f: FlowFieldVal, val):
      return tf.nest.map_structure(lambda x: val * tf.ones_like(x), f)

    # Convert the chemical formulas of the gas species to RRTM-consistent
    # numerical identifiers.
    if vmr_fields is not None and self._optics_lib.gas_optics_sw is not None:
      vmr_fields = {
          self._optics_lib.gas_optics_sw.idx_gases[k]: v
          for k, v in vmr_fields.items()
      }

    def step_fn(igpt, partial_fluxes):
      sw_optical_props = self._optics_lib.compute_sw_optical_properties(
          pressure,
          temperature,
          molecules,
          igpt,
          vmr_fields=vmr_fields,
          cloud_r_eff_liq=cloud_r_eff_liq,
          cloud_path_liq=cloud_path_liq,
          cloud_r_eff_ice=cloud_r_eff_ice,
          cloud_path_ice=cloud_path_ice,
      )
      optical_props_2stream = self._monochrom_solver.sw_cell_properties(
          zenith=self._zenith,
          **sw_optical_props,
      )
      sfc_albedo = tf.nest.map_structure(
          lambda x: self._sfc_albedo * tf.ones_like(x),
          self._rte_utils.slice(
              sw_optical_props['optical_depth'], self._g_dim, 0, 0
          ),
      )
      # Monochromatic top of atmosphere flux.
      solar_flux = self._total_solar_irrad * self._solar_fraction_by_gpt[igpt]
      toa_flux = field_like(sfc_albedo, val=solar_flux)

      sources_2stream = self._monochrom_solver.sw_cell_source(
          replica_id,
          replicas,
          t_dir=optical_props_2stream['t_dir'],
          r_dir=optical_props_2stream['r_dir'],
          optical_depth=sw_optical_props['optical_depth'],
          toa_flux=toa_flux,
          sfc_albedo_direct=sfc_albedo,
          zenith=self._zenith,
      )
      sw_fluxes = self._monochrom_solver.sw_transport(
          replica_id,
          replicas,
          t_diff=optical_props_2stream['t_diff'],
          r_diff=optical_props_2stream['r_diff'],
          src_up=sources_2stream['src_up'],
          src_down=sources_2stream['src_down'],
          sfc_src=sources_2stream['sfc_src'],
          sfc_albedo=sfc_albedo,
          flux_down_dir=sources_2stream['flux_down_dir'],
      )
      total_sw_fluxes = tf.nest.map_structure(
          tf.math.add, sw_fluxes, partial_fluxes
      )
      return igpt + 1, total_sw_fluxes

    stop_condition = lambda i, states: i < self._optics_lib.n_gpt_sw

    fluxes_0 = {
        k: tf.nest.map_structure(tf.zeros_like, pressure)
        for k in ['flux_up', 'flux_down', 'flux_net']
    }
    if self._zenith >= 0.5 * np.pi:
      return fluxes_0
    else:
      i0 = tf.constant(0)
      _, fluxes = tf.nest.map_structure(
          tf.stop_gradient,
          tf.while_loop(
              cond=stop_condition,
              body=step_fn,
              loop_vars=(i0, fluxes_0),
              parallel_iterations=self._optics_lib.n_gpt_sw,
          ),
      )
      return fluxes

  def compute_heating_rate(
      self,
      flux_net: FlowFieldVal,
      pressure: FlowFieldVal,
  ) -> FlowFieldVal:
    """Computes cell-center heating rate from pressure and net radiative flux.

    The net radiative flux corresponds to the bottom cell face. The difference
    of the net flux at the top face and that at the bottom face gives the total
    net flux out of the grid cell. Using the pressure difference across the grid
    cell, the net flux can be converted to a heating rate, in K/s.

    Args:
      flux_net: The net flux at the bottom face [W/m²].
      pressure: The pressure field [Pa].

    Returns:
      The heating rate of the grid cell [K/s].
    """
    # Pressure difference across the atmospheric grid cell.
    dp = tf.nest.map_structure(
        lambda dp_: dp_ / 2.0, self._grad_central(pressure)
    )

    def heating_rate_fn(dflux: tf.Tensor, dp: tf.Tensor):
      """Computes the heating rate at the grid cell center [W]."""
      return constants.G * dflux / dp / constants.CP

    return tf.nest.map_structure(
        heating_rate_fn,
        self._grad_forward_fn(flux_net),
        dp)
