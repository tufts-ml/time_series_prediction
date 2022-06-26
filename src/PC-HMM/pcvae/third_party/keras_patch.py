# Original License for network.py:
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# ==============================================================================
# pylint: disable=protected-access

import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest
from tensorflow.python.util import tf_inspect
import copy



def _make_node_key(layer_name, node_index):
  return layer_name + '_ib-' + str(node_index)

class ConsistantKModel(tf.keras.Model):
    def _insert_layers(self, layers, relevant_nodes=None):
        """Inserts Layers into the Network after Network creation.
        This is only valid for Keras Graph Networks.  Layers added via this function
        will be included in the `call` computation and `get_config` of this Network.
        They will not be added to the Network's outputs.
        Arguments:
          layers: Arbitrary nested structure of Layers. Layers must be reachable
            from one or more of the `keras.Input` Tensors that correspond to this
            Network's inputs.
          relevant_nodes: Nodes from the Layers that should be considered part of
            this Network. If `None`, all Nodes will be considered part of this
            Network.
        Raises:
          ValueError: If the layers depend on `Input`s not found in this Model.
        """
        layers = nest.flatten(layers)
        tf_utils.assert_no_legacy_layers(layers)
        node_to_depth = {}
        for depth, nodes in self._nodes_by_depth.items():
            node_to_depth.update({node: depth for node in nodes})
        # The nodes of these Layers that are relevant to this Network. If not
        # provided, assume all Nodes are relevant
        if not relevant_nodes:
            relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
        network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

        def _get_min_depth(node):
            """Gets the minimum depth at which node can be computed."""
            min_depth = 0
            for layer, node_id, _, _ in node.iterate_inbound():
                inbound_node = layer._inbound_nodes[node_id]
                if inbound_node in node_to_depth:
                    min_depth = min(min_depth, node_to_depth[inbound_node])
                elif inbound_node not in network_nodes:
                    continue
                else:
                    # Previous relevant nodes haven't been processed yet.
                    return None
            # New node is one shallower than its shallowest input.
            return min_depth - 1

        # Insert nodes into `_nodes_by_depth` and other node attrs.
        unprocessed_nodes = copy.copy(relevant_nodes)
        i = 0
        while unprocessed_nodes:
            i += 1
            # Do a sanity check. This can occur if `Input`s from outside this Model
            # are being relied on.
            if i > 1000000:
                raise ValueError('Layers could not be added due to missing '
                                 'dependencies.')

            node = unprocessed_nodes.pop(0)
            depth = _get_min_depth(node)
            if depth is None:  # Defer until inbound nodes are processed.
                unprocessed_nodes.append(node)
                continue
            node_key = _make_node_key(node.layer.name,
                                      node.layer._inbound_nodes.index(node))
            if node_key not in self._network_nodes:
                node_to_depth[node] = depth
                self._network_nodes.add(node_key)
                self._nodes_by_depth[depth].append(node)

        # Insert layers and update other layer attrs.
        layer_set = set(self._layers)
        deferred_layers = []
        for layer in layers:
            if layer not in layer_set:
                self._layers.append(layer)
                deferred_layers.append(layer)
                self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
                layer_set.add(layer)
        self._handle_deferred_layer_dependencies(deferred_layers)

        self._compute_tensor_usage_count()

