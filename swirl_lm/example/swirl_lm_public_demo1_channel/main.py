import functools
import logging
import os
import sys
import time

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from swirl_lm.base import initializer
from swirl_lm.base import parameters as parameters_lib
from swirl_lm.base import driver
from swirl_lm.base import driver_tpu
from swirl_lm.base import parameters
from swirl_lm.physics.combustion import turbulent_kinetic_energy
from swirl_lm.utility import tpu_util

FLAGS = flags.FLAGS

# Setting the precision to float32.
#os.environ["XLA_FLAGS"] = '--xla_jf_conv_full_precision=true'


_TPU = flags.DEFINE_string('tpu', None, 'TPU id.', required=True)
_ZONE = flags.DEFINE_string('zone', None, 'Zone.', required=True)
_PROJECT = flags.DEFINE_string('project', None, 'Cloud project id.', 
                               required=True)

_INIT_FN = flags.DEFINE_string('init_fn', None, 'Init fn.', required=True)
_OUTPUT = flags.DEFINE_string('output', None, 'Output image filename.', 
                              required=True)

flags.DEFINE_float(
    'u_mag',
    1.0,
    'The magnitude of the velocity component in dim 0.',
    allow_override=True)
flags.DEFINE_float(
    'p_ref',
    0.0,
    'The reference pressure used in pressure-induced Taylor-Green vortex.')
flags.DEFINE_float(
    'rho_ref',
    1.0,
    'The reference density used in pressure-induced Taylor-Green vortex.')


# Defines the function that initializes state variables.

def init_fn_channel(replica_id, coordinates):
  """Initializes state variables in a channel flow."""
  del coordinates

  return {
      'replica_id': replica_id,
      'rho': tf.ones((FLAGS.nz, FLAGS.nx, FLAGS.ny), dtype=tf.float32),
      'u': tf.ones((FLAGS.nz, FLAGS.nx, FLAGS.ny), dtype=tf.float32),
      'v': tf.zeros((FLAGS.nz, FLAGS.nx, FLAGS.ny), dtype=tf.float32),
      'w': tf.zeros((FLAGS.nz, FLAGS.nx, FLAGS.ny), dtype=tf.float32),
      'p': tf.zeros((FLAGS.nz, FLAGS.nx, FLAGS.ny), dtype=tf.float32),
   }


def taylor_green_vortices(
    config,
    replica_id,
    coordinates,
):
  """Initialize the u, v, w, and p field in each TPU replica.

  The velocity and pressure fields are initialized following the reference:

  J. R. DeBonis, Solutions of the Taylor-Green vortex problem using
  high-resolution explicit finite difference methods, 51st AIAA Aerospace
  Sciences Meeting including the New Horizons Forum and Aerospace Exposition,
  2013.

  Args:
    replica_id: The ID number of the replica.
    coordinates: A tuple that specifies the replica's grid coordinates in
      physical space.

  Returns:
    A dictionary of states and values that are stored as string and 3D tensor
    pairs.
  """

  v0 = FLAGS.u_mag
  p0 = FLAGS.p_ref
  rho0 = FLAGS.rho_ref

  def get_vortices(state_key):
    """Generates the vortex field for each flow variable."""

    def get_u(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the velocity component in dim 0.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: The sub-mesh in dimension 1 in the present replica.
        zz: The sub-mesh in dimension 2 in the present replica.
        lx: Length in dimension 0.
        ly: Length in dimension 1.
        lz: Length in dimension 2.
        coord: The coordinate of the local core.

      Returns:
        The 3D velocity field in dimension 0 in the present replica.
      """
      del coord
      x_corr = config.dx / (lx + config.dx) * 2.0 * np.pi
      y_corr = config.dy / (ly + config.dy) * 2.0 * np.pi
      z_corr = config.dz / (lz + config.dz) * 2.0 * np.pi
      return v0 * tf.math.sin((2.0 * np.pi - x_corr) * xx / lx) * tf.math.cos(
          (2.0 * np.pi - y_corr) * yy / ly) * tf.math.cos(
              (2.0 * np.pi - z_corr) * zz / lz)

    def get_v(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the velocity component in dim 1.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: The sub-mesh in dimension 1 in the present replica.
        zz: The sub-mesh in dimension 2 in the present replica.
        lx: Length in dimension 0.
        ly: Length in dimension 1.
        lz: Length in dimension 2.
        coord: The coordinate of the local core.

      Returns:
        The 3D velocity field in dimension 1 in the present replica.
      """
      del coord
      x_corr = config.dx / (lx + config.dx) * 2.0 * np.pi
      y_corr = config.dy / (ly + config.dy) * 2.0 * np.pi
      z_corr = config.dz / (lz + config.dz) * 2.0 * np.pi
      return -v0 * tf.math.cos(
          (2.0 * np.pi - x_corr) * xx / lx) * tf.math.sin(
              (2.0 * np.pi - y_corr) * yy / ly) * tf.math.cos(
                  (2.0 * np.pi - z_corr) * zz / lz)

    def get_w(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the velocity component in dim 2.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: Not used.
        zz: Not used.
        lx: Not used.
        ly: Not used.
        lz: Not used.
        coord: The coordinate of the local core.

      Returns:
        The 3D velocity field in dimension 2 in the present replica.
      """
      del yy, zz, lx, ly, lz, coord
      return tf.zeros_like(xx, dtype=tf.float32)

    def get_p(
        xx,
        yy,
        zz,
        lx,
        ly,
        lz,
        coord,
    ):
      """Generates the pressure field.

      Args:
        xx: The sub-mesh in dimension 0 in the present replica.
        yy: The sub-mesh in dimension 1 in the present replica.
        zz: The sub-mesh in dimension 2 in the present replica.
        lx: Length in dimension 0.
        ly: Length in dimension 1.
        lz: Length in dimension 2.
        coord: The coordinate of the local core.

      Returns:
        The 3D pressure field in the present replica.
      """
      del coord
      x_corr = config.dx / (lx + config.dx) * 2.0 * np.pi
      y_corr = config.dy / (ly + config.dy) * 2.0 * np.pi
      z_corr = config.dz / (lz + config.dz) * 2.0 * np.pi
      return p0 + rho0 * v0**2 / 16.0 * (
          (tf.math.cos(2.0 * (2.0 * np.pi - z_corr) * zz / lz) + 2.) *
          (tf.math.cos(2.0 * (2.0 * np.pi - x_corr) * xx / lx) +
           tf.math.cos(2.0 * (2.0 * np.pi - y_corr) * yy / ly)))

    if state_key == 'u':
      return get_u
    elif state_key == 'v':
      return get_v
    elif state_key == 'w':
      return get_w
    elif state_key == 'p':
      return get_p
    else:
      raise ValueError(
          'State key must be one of u, v, w, p. {} is given.'.format(
              state_key))

  output = {'replica_id': replica_id}
  _FIELDS = ('u', 'v', 'w', 'p')

  for key in _FIELDS:
    output.update({
        key:
            initializer.partial_mesh_for_core(
                config,
                coordinates,
                get_vortices(key),
                num_boundary_points=0)
    })

  if (config.solver_procedure ==
      parameters_lib.SolverProcedure.VARIABLE_DENSITY):
    output.update({'rho': tf.ones_like(output['u'], dtype=tf.float32)})

  return output


  # Utility functions for postprocessing.

def merge_result(values, coordinates, halo_width):
  """Merges results from multiple TPU replicas following the topology."""
  if len(values) != len(coordinates):
    raise(
        ValueError,
        f'The length of `value` and `coordinates` must equal. Now `values` has '
        f'length {len(values)}, but `coordinates` has length '
        f'{len(coordinates)}.')

  # The results are oriented in order z-x-y.
  nz, nx, ny = values[0].shape
  nz_0, nx_0, ny_0 = [n - 2 * halo_width for n in (nz, nx, ny)]

  # The topology is oriented in order x-y-z.
  cx, cy, cz = np.array(np.max(coordinates, axis=0)) + 1

  # Compute the total size without ghost cells/halos.
  shape = [n * c for n, c in zip([nz_0, nx_0, ny_0], [cz, cx, cy])]

  result = np.empty(shape, dtype=np.float32)

  for replica in range(len(coordinates)):
    s = np.roll(
        [c * n for c, n in zip(coordinates[replica], (nx_0, ny_0, nz_0))],
        shift=1)
    e = [s_i + n for s_i, n in zip(s, (nz_0, nx_0, ny_0))]
    result[s[0]:e[0], s[1]:e[1], s[2]:e[2]] = (
        values[replica][halo_width:nz_0 + halo_width,
                        halo_width:nx_0 + halo_width,
                        halo_width:ny_0 + halo_width])

  return result


def main(unused_argv):
  # Prepares the simulation configuration.
  params = parameters.params_from_config_file_flag()
  computation_shape = np.array([params.cx, params.cy, params.cz])

  tpu = tf.distribute.cluster_resolver.TPUClusterResolver(
    tpu=_TPU.value, zone=_ZONE.value, project=_PROJECT.value
  ) 
  tf.config.experimental_connect_to_cluster(tpu)
  tf.tpu.experimental.initialize_tpu_system(tpu)
  topology = tf.tpu.experimental.initialize_tpu_system(tpu)
  device_assignment, _ = tpu_util.tpu_device_assignment(
        computation_shape=computation_shape, tpu_topology=topology)
  tpu_strategy = tf.distribute.experimental.TPUStrategy(
      tpu, device_assignment=device_assignment)
  logical_coordinates = tpu_util.grid_coordinates(computation_shape).tolist()

  print("All devices: ", tf.config.list_logical_devices('TPU'))

  init_fn_map = {
    'channel': init_fn_channel,
    'hit': functools.partial(taylor_green_vortices, params)
  }

  init_fn = init_fn_map.get(_INIT_FN.value)
  assert init_fn is not None, (
    f'init_fn {_INIT_FN.value} not one of {init_fn_map.keys()}')
  # initializes the simulation.
  state = driver_tpu.distribute_values(
    tpu_strategy, value_fn=init_fn,
    logical_coordinates=logical_coordinates)

  # Runs the simulation for one cycle.
  step_id = tf.constant(0)

  state = driver._one_cycle(
      strategy=tpu_strategy,
      init_state=state,
      init_step_id=step_id,
      num_steps=FLAGS.num_steps,
      params=params)

  step_id += FLAGS.num_steps

  start_time_s = time.perf_counter()
  state = driver._one_cycle(
      strategy=tpu_strategy,
      init_state=state,
      init_step_id=step_id,
      num_steps=FLAGS.num_steps,
      params=params)
  duration_s = time.perf_counter() - start_time_s
  logging.info('%s steps in %f secs, avg secs/step: %f',
               FLAGS.num_steps, duration_s, duration_s / FLAGS.num_steps)
  
  varname = 'v'  # ['u', 'v', 'w', 'p', 'rho']

  result = merge_result(
      state[varname].values, logical_coordinates, FLAGS.halo_width)

  nx = (FLAGS.nx - 2 * FLAGS.halo_width) * FLAGS.cx
  ny = (FLAGS.ny - 2 * FLAGS.halo_width) * FLAGS.cy
  nz = (FLAGS.nz - 2 * FLAGS.halo_width) * FLAGS.cz

  x = np.linspace(0.0, FLAGS.lx, nx)
  y = np.linspace(0.0, FLAGS.ly, ny)
  z = np.linspace(0.0, FLAGS.lz, nz)

  fig, ax = plt.subplots(figsize=(18, 6))
  c = ax.contourf(x, y, result[nz // 2, ...].transpose(), cmap='jet', 
                  levels=21)
  fig.colorbar(c)
  ax.axis('equal')
  fig.savefig(_OUTPUT.value)


if __name__ == '__main__':
  app.run(main)


