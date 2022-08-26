import os
import sys

from absl import app
from absl import flags
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

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

_OUTPUT = flags.DEFINE_string('output', None, 'Output image filename.', 
                              required=True)


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

  # initializes the simulation.
  state = driver_tpu.distribute_values(
    tpu_strategy, value_fn=init_fn_channel,
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

  state = driver._one_cycle(
      strategy=tpu_strategy,
      init_state=state,
      init_step_id=step_id,
      num_steps=FLAGS.num_steps,
      params=params)
  
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


