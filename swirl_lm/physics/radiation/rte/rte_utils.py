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
"""Utility library for solving the radiative transfer equation (RTE)."""

import inspect
from typing import Callable, List, Sequence, Tuple

import numpy as np
from swirl_lm.communication import halo_exchange
from swirl_lm.utility import common_ops
from swirl_lm.utility import grid_parametrization
from swirl_lm.utility import types
import tensorflow as tf


FlowFieldVal = types.FlowFieldVal
FlowFieldMap = types.FlowFieldMap


class RTEUtils:
  """A library for distributing radiative transfer computations on TPU's.

  Attributes:
    params: An instance of `GridParametrization` containing the grid dimensions
    and information about the TPU computational topology.
    num_cores: A 3-tuple containing the number of cores assigned to each
      dimension.
    grid_size: The local grid dimensions per core.
    halos: The number of halo points on each face of the grid.
  """

  def __init__(
      self,
      params: grid_parametrization.GridParametrization,
  ):
    self.params = params
    self.num_cores = (params.cx, params.cy, params.cz)
    self.grid_size = (params.nx, params.ny, params.nz)
    self.halos = params.halo_width

  def slice(
      self,
      f: types.FlowFieldVal,
      dim: int,
      idx: int,
      face: int,
  ) -> FlowFieldVal:
    """Slices a plane from `f` normal to `dim`."""
    face_slice = common_ops.get_face(f, dim, face, idx)
    if isinstance(f, tf.Tensor) or dim != 2:
      # Remove the outer list.
      return face_slice[0]
    return face_slice

  def _append(
      self,
      a: FlowFieldVal,
      b: FlowFieldVal,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Appends `a` to `b` along `axis` if `forward` and `b` to `a` otherwise."""
    if not forward:
      a, b = b, a
    a_is_tensor = isinstance(a, tf.Tensor)
    assert a_is_tensor == isinstance(b, tf.Tensor)

    if not a_is_tensor and dim == 2:  # Append to Python list.
      return a + b

    if a_is_tensor:
      # Handles the case of single 3D tensor. Shifts `dim` to conform with the
      # 2-0-1 3D tensor orientation.
      axis = (dim + 1) % 3
    else:
      axis = dim

    return tf.nest.map_structure(
        lambda x, y: tf.concat([x, y], axis=axis), a, b
    )

  def _pad(
      self,
      f: FlowFieldVal,
      low_n: int,
      high_n: int,
      dim: int
  ) -> FlowFieldVal:
    """Pads the field with zeros along the dimension `dim`."""
    paddings = [(0, 0)] * 3
    paddings[dim] = (low_n, high_n)
    return common_ops.pad(f, paddings)

  def _generate_adjacent_pair_assignments(
      self, replicas: np.ndarray, axis: int, forward: bool
  ) -> List[np.ndarray]:
    """Creates groups of source-target TPU devices along `axis`.

    The group assignments are used by `tf.raw.CollectivePermute` to exchange
    data between neighboring replicas along `axis`. There will be one such group
    for every interface of the topology. As an example consider a `replicas` of
    shape `[4, 2, 1]` where the replica id's are 0 through 7. Each replica is
    assigned to a unique 3-tuple `coordinate`. In this example, the mapping of
    coordinates to replica ids is `{(0, 0, 0): 0, (0, 1, 0): 1, (1, 0, 0): 2,
    (1, 1, 0): 3, (2, 0, 0): 4, (2, 1, 0): 5, (3, 0, 0): 6, (3, 1, 0): 7}` with
    each element following the form `(coordinate[0], coordinate[1],
    coordinate[2]): replica_id`. The group assignment along dimension 0 contains
    the replica ids with the same coordinate of dimensions 1 and 2 that neighbor
    each other along `axis` in the direction of increasing index if `forward` is
    set to `True`, and in the direction of decreasing index otherwise. In this
    example, if `axis` is 0 and `forward` is `True`, the assignments are:

    [[[0, 2], [1, 3]],
     [[[2, 4], [3, 5]],
     [[[4, 6], [5, 7]]].

    If `forward` is `False`, the pairs for each inteface are reversed:

    [[2, 0], [3, 1]],
     [[4, 2], [5, 3]],
     [[6, 4], [7, 5]]].

    Args:
      replicas: The mapping from the core coordinate to the local replica id.
      axis: The axis along which data will be propagated.
      forward: Whether the data propagation of `tf.raw_ops.CollectivePermute`
        will unravel in the direction of increasing index along `axis`.

    Returns:
      A list of groups of adjacent replica id pairs. There will be one such
      group for every interface of the computational topology along `axis`.
    """
    groups = common_ops.group_replicas(replicas, axis=axis)
    pair_groups = []
    depth = groups.shape[1]
    for i in range(depth - 1):
      pair_group = groups[:, i : i + 2]
      if not forward:
        # Reverse the order of the source-target pairs.
        pair_group = pair_group[:, ::-1]
      pair_groups.append(pair_group.tolist())
    return pair_groups

  def _local_recurrent_op(
      self,
      recurrent_fn: Callable[..., tf.Tensor],
      variables: FlowFieldMap,
      dim: int,
      n: int,
      forward: bool = True,
  ) -> Tuple[FlowFieldVal, FlowFieldVal]:
    """Computes a sequence of recurrent operations along a dimension.

    Each core performs the same operation on data local to it independently.
    Note that the initial input `x0` in `variables` is not included in the final
    output.

    Args:
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence. `x0` has the
        same structure as other fields in `variables` (i.e. either a 3-D
        tf.Tensor or a list of 2D `tf.Tensor`s) but has a size of 1 along the
        axis determined by `dim`.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      n: The number of layers in the final solution.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = recurrent_fn(x[i + 1]).

    Returns:
      A tuple containing 1) A 3D variable with the cumulative output from the
      chain of recurrent transformations having the same structure and shape as
      any field in `variables` that is not `x0`. 2) the 2D output of the last
      recurrent transformation.
    """
    x = variables['x0']

    face = 0 if forward else 1

    for i in range(n):
      prev_idx = i - 1
      plane_args = {
          k: self.slice(v, dim, i, face)
          for k, v in variables.items()
          if k != 'x0'
      }
      plane_args['x0'] = (
          x if i == 0 else self.slice(x, dim, prev_idx, face)
      )
      arg_lst = [
          plane_args[k] for k in inspect.getfullargspec(recurrent_fn).args
      ]
      next_layer = tf.nest.map_structure(recurrent_fn, *arg_lst)
      x = next_layer if i == 0 else self._append(x, next_layer, dim, forward)

    last_local_layer = self.slice(x, dim, n - 1, face)

    return x, last_local_layer

  def _cumulative_recurrent_op_sequential(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., FlowFieldVal],
      variables: FlowFieldMap,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Computes a sequence of recurrent affine transformations globally.

    This particular implementation is sequential, so every layer of replicas
    along the accumulation axis needs to wait for the previous computational
    layer to complete before proceeding, which can be slow. On the other hand,
    this approach has a very small memory overhead, since the TPU communication
    only happens between pairs of adjacent computational layers through the
    `tf.raw_ops.CollectivePermute` operation.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      recurrent_fn: The local cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = w[i] * x[i + 1] + b[i].

    Returns:
      A 3D variable with the cumulative output from the chain of recurrent
      transformations having the same structure and shape as any field in
      `variables` that is not `x0`.
    """
    halos = [0] * 3
    halos[dim] = self.halos

    n = self.grid_size[dim] - 2 * self.halos

    # Remove halos along the axis.
    kwargs = {
        k: common_ops.strip_halos(v, halos)
        for k, v in variables.items()
        if k != 'x0'
    }
    kwargs['x0'] = variables['x0']

    def local_fn(x0: FlowFieldVal) -> Tuple[FlowFieldVal, FlowFieldVal]:
      """Generates the output of a cumulative operation and its last layer."""
      kwargs['x0'] = x0
      return self._local_recurrent_op(recurrent_fn, kwargs, dim, n, forward)

    def communicate_fn(x):
      return tf.raw_ops.CollectivePermute(
          input=x, source_target_pairs=pair_group
      )

    core_idx = common_ops.get_core_coordinate(replicas, replica_id)[dim]

    # Cumulative local output and its last layer. This result will only be valid
    # for the first level of the computational topology. All the subsequent
    # levels will need to wait for the output from the previous level before
    # evaluating their local function.
    x_cum, x_out = local_fn(x0=variables['x0'])

    pair_groups = self._generate_adjacent_pair_assignments(
        replicas, dim, forward
    )
    n_groups = len(pair_groups)
    interface_iter = range(n_groups) if forward else reversed(range(n_groups))

    # Sequentially evaluate a level of the computational topology and propagate
    # results to the next level.
    for i in interface_iter:
      pair_group = pair_groups[i]
      # Send / receive the last recurrent output layer.
      x_prev = tf.nest.map_structure(communicate_fn, x_out)
      # Index of the next set of cores receiving the data.
      recv_core_idx = i + 1 if forward else i
      x_cum, x_out = tf.cond(
          tf.equal(core_idx, recv_core_idx),
          true_fn=lambda: local_fn(x0=x_prev),  # pylint: disable=cell-var-from-loop
          false_fn=lambda: (x_cum, x_out))

    # Pad the result with halo layers.
    return self._pad(x_cum, self.halos, self.halos, dim)

  def cumulative_recurrent_op(
      self,
      replica_id: tf.Tensor,
      replicas: np.ndarray,
      recurrent_fn: Callable[..., FlowFieldVal],
      variables: FlowFieldMap,
      dim: int,
      forward: bool = True,
  ) -> FlowFieldVal:
    """Applies a recurrent operation globally along a specified dimension.

    This global operation is sequential and will process each layer along `dim`
    at a time in increase order if `forward` is `True` and in decreasing order
    otherwise. As a consequence, if there are multiple TPU cores assigned to
    dimension `dim` in the computational topology some cores will experience
    idleness as they wait for previous layers residing in other cores to be
    processed.

    Args:
      replica_id: The index of the current TPU replica.
      replicas: The mapping from the core coordinate to the local replica id
        `replica_id`.
      recurrent_fn: The cumulative recurrent operation.
      variables: A dictionary containing the local fields that will be inputs to
        `recurrent_fn`. One of the entries must be `x0`, which should correspond
        to the boundary solution that initiates the recurrence.
      dim: The physical dimension along which the sequence of affine
        transformations will be applied.
      forward: Whether the accumulation starts with the first layer of
        coefficients. If set to False, then the recurrence relation unravels
        from the last layer to the first as follows:
        x[i] = recurrent_fn({'x0': x[i + 1], ...})

    Returns:
      A 3D variable with the cumulative output from the chain of recurrent
      transformations having the same structure and shape as any field in
      `variables` that is not `x0` and having `x0` as the boundary value at the
      face that initiates the recurrence.
    """
    val = self._cumulative_recurrent_op_sequential(
        replica_id, replicas, recurrent_fn, variables, dim, forward
    )
    x0 = variables['x0']

    # If the boundary condition is set along the list dimension, the halo
    # exchange expects the boundary plane represented as a 2D tensor. Otherwise,
    # if set along one of the tensor dimensions, a list of thin tensors of
    # dimension (1, ny) or (nx, 1) is expected.
    if isinstance(x0, Sequence) and dim == 2:
      x0 = x0[0]

    face = 0 if forward else 1
    bc = [[(halo_exchange.BCType.NEUMANN, 0.0)] * 2 for _ in range(3)]
    # Set the boundary plane that initiates the recurrent operation above as the
    # halo values.
    bc[dim][face] = (
        halo_exchange.BCType.DIRICHLET,
        [x0] * self.halos,
    )
    return halo_exchange.inplace_halo_exchange(
        val,
        (0, 1, 2),
        replica_id,
        replicas,
        (0, 1, 2),
        (False, False, False),
        boundary_conditions=bc,
        width=self.halos,
    )
