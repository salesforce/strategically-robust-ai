from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ray.rllib.models import ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.recurrent_tf_model_v2 import RecurrentNetwork
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils import try_import_tf

from gym.spaces import Box, Dict
import numpy as np

tf = try_import_tf()[0]
from tensorflow import keras


def get_clean_key(key):
    return key.replace(".", "-")


def create_cnn(
    input_space, num_conv_layers=1, num_channels=16, kernel_size=3, stride=2
):

    conv_map_channels, conv_shape_r, conv_shape_c = input_space.shape
    conv_shape = (conv_shape_r, conv_shape_c, conv_map_channels)

    conv_model = keras.models.Sequential(name="conv_model")
    conv_model.add(keras.layers.Permute((2, 3, 1)))
    conv_model.add(
        keras.layers.Conv2D(
            num_channels,
            kernel_size,
            stride,
            activation="relu",
            input_shape=conv_shape,
            name="conv-relu-0",
        )
    )
    for i in range(num_conv_layers - 1):
        conv_model.add(
            keras.layers.Conv2D(
                num_channels,
                kernel_size,
                stride,
                activation="relu",
                name="conv-relu-{}".format(i + 1),
            )
        )
    conv_model.add(keras.layers.Flatten())

    return conv_model


def apply_logit_mask(logits, mask):
    """Mask values of 1 are valid actions. Add huge negative values to logits with 0 mask values."""
    logit_mask = tf.ones_like(logits) * -10000000
    logit_mask = logit_mask * (1 - mask)

    return logits + logit_mask


def recursive_get_nd_features(
    input_space,
    num_conv_layers=1,
    num_channels=16,
    kernel_size=3,
    stride=2,
    blacklist=[],
):
    if isinstance(input_space, Box):
        if len(input_space.shape) == 1:  # assume it must be processed by an FFN
            return None, 0
        elif len(input_space.shape) == 2:  # assume it must be processed by a CNN
            return None, 0
        elif len(input_space.shape) == 3:  # assume it must be processed by a CNN
            cnn = create_cnn(
                input_space,
                num_conv_layers=num_conv_layers,
                num_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
            )
            conv_td = keras.layers.TimeDistributed(cnn)
            return conv_td, 1
        else:
            return None, 0
    elif isinstance(input_space, Dict):
        module_pydict = {}
        num_nd_boxes = 0
        for k, space in input_space.spaces.items():
            if k not in blacklist:
                clean_k = get_clean_key(k)
                feature, _num_nd_boxes = recursive_get_nd_features(
                    space,
                    num_conv_layers=num_conv_layers,
                    num_channels=num_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    blacklist=blacklist,
                )
                if feature is not None:
                    module_pydict[clean_k] = feature
                    num_nd_boxes += _num_nd_boxes
        return module_pydict, num_nd_boxes
    else:
        raise AssertionError


def recursive_get_shapes(input_space, blacklist=[]):
    if isinstance(input_space, Box):
        return input_space.shape
    elif isinstance(input_space, Dict):
        shape_dict = {}
        for k, space in input_space.spaces.items():
            if k not in blacklist:
                shape_dict[k] = recursive_get_shapes(space, blacklist=blacklist)
        return shape_dict
    else:
        raise AssertionError


def recursive_get_boxes(space, blacklist=[]):
    boxes = []
    if isinstance(space, Box):
        boxes.append(space)
    elif isinstance(space, Dict):
        for k, space in space.spaces.items():
            if k not in blacklist:
                boxes.extend(recursive_get_boxes(space, blacklist=blacklist))
    else:
        raise AssertionError
    return boxes


class KerasConvRNN(RecurrentNetwork):
    custom_name = "ppo_conv_lstm_tf"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Convention: [Batch, Time, Agent, *]
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Only use the parameters from the custom_model_config dict!
        custom_model_config = model_config["custom_model_config"]

        input_emb_dim = custom_model_config["input_emb_dim"]
        lstm_cell_size = custom_model_config["lstm_cell_size"]
        fc_dim = custom_model_config["fc_dim"]
        num_fc = custom_model_config["num_fc"]
        num_conv = custom_model_config["num_conv"]
        num_channels = custom_model_config["num_channels"]
        kernel_size = custom_model_config["kernel_size"]
        stride = custom_model_config["stride"]

        # Constants
        self.MASK_NAME = "action_mask"
        self.input_blacklist = [self.MASK_NAME]

        self.lstm_cell_size = lstm_cell_size

        self.values = None  # Model only outputs Policy logits, values are accessed using self.value_function

        # Check Observation spaces
        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        if not isinstance(obs_space, Dict):
            if isinstance(obs_space, Box):
                raise TypeError(
                    "({}) Observation space should be a gym Dict. Is a Box of shape {}".format(
                        name, obs_space.shape
                    )
                )
            else:
                raise TypeError(
                    "({}) Observation space should be a gym Dict. Is {} instead.".format(
                        name, type(obs_space)
                    )
                )

        # === Define input layers ===
        self._input_keys = []
        input_dict = {}
        for k, v in obs_space.spaces.items():
            print(k, v, v.shape)
            shape = (None,) + v.shape
            input_dict[k] = keras.layers.Input(shape=shape, name=k)
            self._input_keys.append(k)

        state_in = [
            keras.layers.Input(shape=(self.lstm_cell_size,), name="h_policy"),
            keras.layers.Input(shape=(self.lstm_cell_size,), name="c_policy"),
            keras.layers.Input(shape=(self.lstm_cell_size,), name="h_value"),
            keras.layers.Input(shape=(self.lstm_cell_size,), name="c_value"),
        ]
        seq_lens_in = keras.layers.Input(shape=(), name="seq_lens")

        # === Start constructing model ===
        self.tags = ["policy", "value"]

        self.sub_modules = {}

        for tag in self.tags:

            sub_modules_for_tag = {}

            # === Input ===
            input_features_dict = {}

            # Define input layers: 1d (concat these)
            boxes = recursive_get_boxes(obs_space, blacklist=self.input_blacklist)
            flat_oned_shape = sum([i.shape[0] for i in boxes if len(i.shape) == 1])
            input_features_dict["flat"] = keras.layers.Dense(
                input_emb_dim, input_shape=(flat_oned_shape,), name="emb-" + tag
            )

            # Define input layers: nd
            nd_features_dict, _ = recursive_get_nd_features(
                obs_space,
                num_conv_layers=num_conv,
                num_channels=num_channels,
                kernel_size=kernel_size,
                stride=stride,
                blacklist=self.input_blacklist,
            )
            input_features_dict.update(nd_features_dict)

            sub_modules_for_tag["input_features"] = input_features_dict

            # === Middle: stack of FC layers ===
            middle_fc_dims = [fc_dim] * num_fc

            middle_fc = keras.Sequential(name=tag)
            for i in range(num_fc):
                middle_fc.add(
                    keras.layers.Dense(
                        middle_fc_dims[i], name="middle-fc-relu-{}".format(i)
                    )
                )
            middle_fc.add(keras.layers.LayerNormalization(axis=-1, name="middle-ln"))

            sub_modules_for_tag["middle"] = middle_fc

            # === LSTM ===
            sub_modules_for_tag["lstm"] = keras.layers.LSTM(
                self.lstm_cell_size,
                return_sequences=True,
                return_state=True,
                name="lstm-" + tag,
            )

            # === Output ===
            if tag == "policy":
                assert self.MASK_NAME in obs_space.spaces
                assert (
                    isinstance(obs_space.spaces[self.MASK_NAME], Box)
                    and len(obs_space.spaces[self.MASK_NAME].shape) == 1
                )
                total_num_logits = obs_space.spaces[self.MASK_NAME].shape[0]
                print(
                    "total_num_logits == self.num_outputs",
                    total_num_logits,
                    self.num_outputs,
                )
                assert total_num_logits == self.num_outputs

                sub_modules_for_tag["output"] = keras.layers.Dense(
                    total_num_logits,
                    input_shape=(self.lstm_cell_size,),
                    name="dense-logits",
                )
            elif tag == "value":
                sub_modules_for_tag["output"] = keras.layers.Dense(
                    1, input_shape=(self.lstm_cell_size,), name="dense-values"
                )
            else:
                raise ValueError

            self.sub_modules[tag] = sub_modules_for_tag

        self.input_dict = input_dict

        # === Passing through the model ===
        lstm_states = {
            self.tags[0]: [state_in[0], state_in[1]],
            self.tags[1]: [state_in[2], state_in[3]],
        }
        new_lstm_states = {tag: None for tag in self.tags}

        masked_logits = None

        for tag in self.tags:

            sub_module = self.sub_modules[tag]

            vector_obs = []
            input_features = {}

            # Process nd inputs, gather 1d inputs
            for k, input_tensor_with_time in input_dict.items():
                if k in self.input_blacklist:
                    pass
                elif len(input_tensor_with_time.shape) == 4:
                    # Reshape (B, T, dim1, dim2) -> (B, T, dim1*dim2)
                    B = tf.shape(input_tensor_with_time)[0]
                    T = tf.shape(input_tensor_with_time)[1]
                    dim1 = input_tensor_with_time.shape[2]
                    dim2 = input_tensor_with_time.shape[3]
                    input_tensor_with_time = tf.reshape(input_tensor_with_time, [B, T, dim1*dim2])
                    vector_obs.append(input_tensor_with_time)
                else:
                    assert isinstance(input_tensor_with_time, tf.Tensor)

                    if len(input_tensor_with_time.shape) == 3:  # [B, T, F]
                        vector_obs.append(input_tensor_with_time)
                    else:
                        clean_k = get_clean_key(k)

                        featurizer = sub_module["input_features"][clean_k]

                        nd_feature = featurizer(input_tensor_with_time)

                        _bsz, _nt = nd_feature.shape[0], nd_feature.shape[1]

                        flat_nd_feature = keras.layers.TimeDistributed(
                            keras.layers.Flatten()
                        )(nd_feature)

                        input_features[clean_k] = flat_nd_feature

            # Concat 1d inputs and featurize
            concat_vector_obs = tf.concat(vector_obs, axis=-1)
            input_features["flat"] = sub_module["input_features"]["flat"](
                concat_vector_obs
            )

            # All features are 1d (+ batch + time)
            for k, feature in input_features.items():
                assert len(feature.shape) == 3  # [B, T, F]

            # Concat all features
            concat_all_input_features = tf.concat(
                [v for k, v in input_features.items()], axis=-1
            )

            # Middle FC
            middle_fc_features = sub_module["middle"](concat_all_input_features)

            # LSTM
            state_h, state_c = lstm_states[tag][0], lstm_states[tag][1]

            lstm_output, new_state_h, new_state_c = sub_module["lstm"](
                middle_fc_features, (state_h, state_c)
            )

            new_lstm_states[tag] = [new_state_h, new_state_c]

            # FC - final
            if tag == "policy":
                unmasked_logits = sub_module["output"](lstm_output)
                masked_logits = apply_logit_mask(
                    unmasked_logits, input_dict[self.MASK_NAME]
                )
            elif tag == "value":
                values = sub_module["output"](lstm_output)

        assert masked_logits is not None

        prev_states = lstm_states[self.tags[0]] + lstm_states[self.tags[1]]
        states = new_lstm_states[self.tags[0]] + new_lstm_states[self.tags[1]]

        # === Keras model definition ===
        self.model = keras.Model(
            inputs=self._extract_input_list(input_dict) + [seq_lens_in, prev_states],
            outputs=[masked_logits, values, *states],
        )
        self.register_variables(self.model.variables)

        print("=" * 40)
        print(
            "Created Model:\n{} \nCUDA: {}".format(
                self.model.summary(), "TODO: FIX THIS"
            )
        )
        print("=" * 40)

    def get_initial_state(self):
        return [
            np.zeros(self.lstm_cell_size, np.float32),
            np.zeros(self.lstm_cell_size, np.float32),
            np.zeros(self.lstm_cell_size, np.float32),
            np.zeros(self.lstm_cell_size, np.float32),
        ]

    def _extract_input_list(self, dictionary):
        return [dictionary[k] for k in self._input_keys]

    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        masked_logits, states = self.forward_rnn(
            [
                add_time_dimension(t, seq_lens)
                for t in self._extract_input_list(input_dict["obs"])
            ],
            state,
            seq_lens,
        )
        return tf.reshape(masked_logits, (-1, self.num_outputs)), states

    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self.values, h_policy, c_policy, h_value, c_value = self.model(
            inputs + [seq_lens] + state
        )
        return model_out, [h_policy, c_policy, h_value, c_value]

    def value_function(self):
        return tf.reshape(self.values, [-1])


ModelCatalog.register_custom_model("ppo_conv_lstm_tf", KerasConvRNN)


class RandomAction(TFModelV2):
    """Custom model for policy gradient algorithms."""

    custom_name = "random"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        if hasattr(obs_space, "original_space"):
            original_space = obs_space.original_space
        else:
            assert isinstance(obs_space, Dict)
            original_space = obs_space

        self.MASK_NAME = "action_mask"
        mask = original_space.spaces[self.MASK_NAME]
        mask_input = keras.layers.Input(shape=mask.shape, name=self.MASK_NAME)

        self.inputs = [
            keras.layers.Input(shape=(1,), name="observations"),
            mask_input,
        ]

        logits_and_value = keras.layers.Dense(
            num_outputs + 1, activation=None, name="dummy_layer"
        )(self.inputs[0])

        unmasked_logits = logits_and_value[:, :num_outputs] * 0.0
        values = logits_and_value[:, -1]

        masked_logits = apply_logit_mask(unmasked_logits, mask_input)

        self.base_model = keras.Model(self.inputs, [masked_logits, values])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self.values = self.base_model(
            [input_dict["obs_flat"][:, :1], input_dict["obs"][self.MASK_NAME]]
        )
        return model_out, state

    def value_function(self):
        return tf.reshape(self.values, [-1])


ModelCatalog.register_custom_model("random", RandomAction)

class KerasFC(TFModelV2):
    """Custom model for policy gradient algorithms."""

    custom_name = "ppo_fc"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = [
            tf.keras.layers.Input(
                shape=(get_flat_obs_size(obs_space),), name="observations"
            ),
        ]

        self.MASK_NAME = "action_mask"
        self.use_mask = self.MASK_NAME in obs_space.original_space.spaces
        if self.use_mask:
            mask = obs_space.original_space.spaces[self.MASK_NAME]
            mask_input = tf.keras.layers.Input(shape=mask.shape, name=self.MASK_NAME)
            self.inputs.append(mask_input)

        custom_model_config = model_config["custom_model_config"]

        shared_layer = self.inputs[0]
        if custom_model_config["shared_n"]:
            for i, n in enumerate(custom_model_config["shared_n"]):
                shared_layer = tf.keras.layers.Dense(
                    n, activation=tf.keras.activations.relu, name="shared" + str(i)
                )(shared_layer)

        pre_logits = shared_layer
        if custom_model_config["pre_logits_n"]:
            for i, n in enumerate(custom_model_config["pre_logits_n"]):
                pre_logits = tf.keras.layers.Dense(
                    n, activation=tf.keras.activations.relu, name="pre_logits" + str(i)
                )(pre_logits)

        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(pre_logits)

        if self.use_mask:
            logits = apply_logit_mask(logits, mask_input)

        pre_values = shared_layer
        if custom_model_config["pre_values_n"]:
            for i, n in enumerate(custom_model_config["pre_values_n"]):
                pre_values = tf.keras.layers.Dense(
                    n, activation=tf.keras.activations.relu, name="pre_values" + str(i)
                )(pre_values)

        values = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.linear, name="values"
        )(pre_values)

        self.base_model = tf.keras.Model(self.inputs, [logits, values])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        if self.use_mask:
            model_out, self._value_out = self.base_model(
                [input_dict["obs_flat"], input_dict["obs"][self.MASK_NAME]]
            )
        else:
            model_out, self._value_out = self.base_model(
                [input_dict["obs_flat"]]
            )
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model("ppo_fc", KerasFC)


class KerasLinear(TFModelV2):
    """Custom model for policy gradient algorithms."""

    custom_name = "ppo_linear"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.inputs = [
            tf.keras.layers.Input(
                shape=(get_flat_obs_size(obs_space),), name="observations"
            ),
        ]

        self.MASK_NAME = "action_mask"
        self.use_mask = self.MASK_NAME in obs_space.original_space.spaces
        if self.use_mask:
            mask = obs_space.original_space.spaces[self.MASK_NAME]
            mask_input = tf.keras.layers.Input(shape=mask.shape, name=self.MASK_NAME)
            self.inputs.append(mask_input)

        logits = tf.keras.layers.Dense(
            self.num_outputs, activation=tf.keras.activations.linear, name="logits"
        )(self.inputs[0])

        if self.use_mask:
            logits = apply_logit_mask(logits, mask_input)

        values = tf.keras.layers.Dense(
            1, activation=tf.keras.activations.linear, name="values"
        )(self.inputs[0])

        self.base_model = tf.keras.Model(self.inputs, [logits, values])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        if self.use_mask:
            model_out, self._value_out = self.base_model(
                [input_dict["obs_flat"], input_dict["obs"][self.MASK_NAME]]
            )
        else:
            model_out, self._value_out = self.base_model(
                [input_dict["obs_flat"]]
            )
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model("ppo_linear", KerasLinear)


def get_flat_obs_size(obs_space):
    if isinstance(obs_space, Box):
        return np.prod(obs_space.shape)
    elif not isinstance(obs_space, Dict):
        raise TypeError

    def rec_size(obs_dict_space, n=0):
        for subspace in obs_dict_space.spaces.values():
            if isinstance(subspace, Box):
                n = n + np.prod(subspace.shape)
            elif isinstance(subspace, Dict):
                n = rec_size(subspace, n=n)
            else:
                raise TypeError
        return n

    return rec_size(obs_space)


WORLD_MAP_NAME = "world-map"
WORLD_IDX_MAP_NAME = "world-idx_map"
MASK_NAME = "action_mask"
# GENERIC_NAME = "flat"


class OldStyleKerasConvRNN(RecurrentNetwork):
    """Example of using the Keras functional API to define a RNN model."""

    custom_name = "ppo_conv_lstm_old_style"

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        input_emb_vocab = self.model_config["custom_model_config"]["input_emb_vocab"]
        emb_dim = self.model_config["custom_model_config"]["idx_emb_dim"]
        num_conv = self.model_config["custom_model_config"]["num_conv"]
        num_fc = self.model_config["custom_model_config"]["num_fc"]
        fc_dim = self.model_config["custom_model_config"]["fc_dim"]
        cell_size = self.model_config["custom_model_config"]["lstm_cell_size"]

        generic_name = self.model_config["custom_model_config"].get("generic_name", None)

        self.cell_size = cell_size

        if hasattr(obs_space, "original_space"):
            obs_space = obs_space.original_space

        if not isinstance(obs_space, Dict):
            if isinstance(obs_space, Box):
                raise TypeError(
                    "({}) Observation space should be a gym Dict. Is a Box of shape {}".format(
                        name, obs_space.shape
                    )
                )
            else:
                raise TypeError(
                    "({}) Observation space should be a gym Dict. Is {} instead.".format(
                        name, type(obs_space)
                    )
                )

        # Define input layers
        self._input_keys = []
        non_conv_input_keys = []
        input_dict = {}
        conv_shape_r = None
        conv_shape_c = None
        conv_map_channels = None
        conv_idx_channels = None
        found_world_map = False
        found_world_idx = False
        for k, v in obs_space.spaces.items():
            print(k, v, v.shape)
            shape = (None,) + v.shape
            input_dict[k] = tf.keras.layers.Input(shape=shape, name=k)
            self._input_keys.append(k)
            if k == MASK_NAME:
                pass
            elif k == WORLD_MAP_NAME:
                conv_shape_r, conv_shape_c, conv_map_channels = (
                    v.shape[1],
                    v.shape[2],
                    v.shape[0],
                )
                found_world_map = True
            elif k == WORLD_IDX_MAP_NAME:
                conv_idx_channels = v.shape[0] * emb_dim
                found_world_idx = True
            else:
                non_conv_input_keys.append(k)


        state_in_h_p = tf.keras.layers.Input(shape=(cell_size,), name="h_pol")
        state_in_c_p = tf.keras.layers.Input(shape=(cell_size,), name="c_pol")
        state_in_h_v = tf.keras.layers.Input(shape=(cell_size,), name="h_val")
        state_in_c_v = tf.keras.layers.Input(shape=(cell_size,), name="c_val")
        seq_in = tf.keras.layers.Input(shape=(), name="seq_in")

        # Determine which of the inputs are treated as non-conv inputs
        if generic_name is None:
            non_conv_inputs = tf.keras.layers.concatenate(
                [input_dict[k] for k in non_conv_input_keys]
            )
        elif isinstance(generic_name, (tuple, list)):
            non_conv_inputs = tf.keras.layers.concatenate(
                [input_dict[k] for k in generic_name]
            )
        elif isinstance(generic_name, str):
            non_conv_inputs = input_dict[generic_name]
        else:
            raise TypeError

        if found_world_map:
            assert found_world_idx
            use_conv = True
            conv_shape = (
                conv_shape_r, conv_shape_c, conv_map_channels + conv_idx_channels
            )

            conv_input_map = tf.keras.layers.Permute((1, 3, 4, 2))(
                input_dict[WORLD_MAP_NAME]
            )
            conv_input_idx = tf.keras.layers.Permute((1, 3, 4, 2))(
                input_dict[WORLD_IDX_MAP_NAME]
            )

        else:
            assert not found_world_idx
            use_conv = False
            conv_shape = None
            conv_input_map = None
            conv_input_idx = None

        logits, values, state_h_p, state_c_p, state_h_v, state_c_v = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        for tag in ["_pol", "_val"]:
            if tag == "_pol":
                state_in = [state_in_h_p, state_in_c_p]
            elif tag == "_val":
                state_in = [state_in_h_v, state_in_c_v]
            else:
                raise NotImplementedError

            # Apply convolution to the spatial inputs
            if use_conv:
                map_embedding = tf.keras.layers.Embedding(
                    input_emb_vocab, emb_dim,
                    name='embedding'+tag
                )
                conv_idx_embedding = tf.keras.layers.Reshape(
                    (-1, conv_shape_r, conv_shape_c, conv_idx_channels)
                )(map_embedding(conv_input_idx))

                conv_input = tf.keras.layers.concatenate(
                    [conv_input_map, conv_idx_embedding]
                )

                conv_model = tf.keras.models.Sequential(name="conv_model" + tag)
                assert conv_shape
                conv_model.add(
                    tf.keras.layers.Conv2D(
                        16, (3, 3), strides=2, activation="relu",
                        input_shape=conv_shape, name='conv2D_1'+tag
                    )
                )

                for i in range(num_conv - 1):
                    conv_model.add(
                        tf.keras.layers.Conv2D(
                            32, (3, 3), strides=2, activation="relu",
                            name='conv2D_{}{}'.format(i+2, tag)
                        )
                    )

                conv_model.add(tf.keras.layers.Flatten())

                conv_td = tf.keras.layers.TimeDistributed(conv_model)(conv_input)

                # Combine the conv output with the non-conv inputs
                dense = tf.keras.layers.concatenate([conv_td, non_conv_inputs])

            # No spatial inputs provided -- skip any conv steps
            else:
                dense = non_conv_inputs

            # Preprocess observation with hidden layers and send to LSTM cell
            for i in range(num_fc):
                layer = tf.keras.layers.Dense(
                    fc_dim, activation=tf.nn.relu, name="dense{}".format(i + 1) + tag
                )
                dense = layer(dense)

            dense = tf.keras.layers.LayerNormalization(name="layer_norm" + tag)(dense)

            lstm_out, state_h, state_c = tf.keras.layers.LSTM(
                cell_size, return_sequences=True, return_state=True, name="lstm" + tag
            )(inputs=dense, mask=tf.sequence_mask(seq_in), initial_state=state_in)

            # Project LSTM output to logits or value
            output = tf.keras.layers.Dense(
                self.num_outputs if tag == "_pol" else 1,
                activation=tf.keras.activations.linear,
                name="logits" if tag == "_pol" else "value",
            )(lstm_out)

            if tag == "_pol":
                state_h_p, state_c_p = state_h, state_c
                logits = apply_logit_mask(output, input_dict[MASK_NAME])
            elif tag == "_val":
                state_h_v, state_c_v = state_h, state_c
                values = output
            else:
                raise NotImplementedError

        self.input_dict = input_dict

        for out in [logits, values, state_h_p, state_c_p, state_h_v, state_c_v]:
            assert out is not None

        # Create the RNN model
        self.rnn_model = tf.keras.Model(
            inputs=self._extract_input_list(input_dict)
                   + [seq_in, state_in_h_p, state_in_c_p, state_in_h_v, state_in_c_v],
            outputs=[logits, values, state_h_p, state_c_p, state_h_v, state_c_v],
        )
        self.register_variables(self.rnn_model.variables)
        # self.rnn_model.summary()

    def _extract_input_list(self, dictionary):
        return [dictionary[k] for k in self._input_keys]

    def forward(self, input_dict, state, seq_lens):
        """Adds time dimension to batch before sending inputs to forward_rnn().

        You should implement forward_rnn() in your subclass."""
        output, new_state = self.forward_rnn(
            [
                add_time_dimension(t, seq_lens)
                for t in self._extract_input_list(input_dict["obs"])
            ],
            state,
            seq_lens,
        )
        return tf.reshape(output, [-1, self.num_outputs]), new_state

    def forward_rnn(self, inputs, state, seq_lens):
        model_out, self._value_out, h_p, c_p, h_v, c_v = self.rnn_model(
            inputs + [seq_lens] + state
        )
        return model_out, [h_p, c_p, h_v, c_v]

    def get_initial_state(self):
        return [
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
            np.zeros(self.cell_size, np.float32),
        ]

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


ModelCatalog.register_custom_model(
    OldStyleKerasConvRNN.custom_name, OldStyleKerasConvRNN
)
