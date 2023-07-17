import pdb

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Concatenate, Input, AvgPool2D, BatchNormalization
import numpy as np


def print_node(x):
    print(x)
    return x


class DDPGAgentParams:
    def __init__(self):
        # Convolutional part config
        self.conv_layers = 2
        self.conv_kernel_size = 5
        self.conv_kernels = 16

        # Fully Connected config
        self.hidden_layer_size = 256
        self.hidden_layer_num = 3

        # Training Params
        self.learning_rate = 3e-5
        self.alpha = 0.005
        self.gamma = 0.95

        # Exploration strategy
        self.soft_max_scaling = 0.1

        # Global-Local Map
        self.global_map_scaling = 3
        self.local_map_size = 17

        # Scalar inputs instead of map
        self.use_scalar_input = False
        self.relative_scalars = False
        self.blind_agent = False
        self.max_uavs = 3
        self.max_devices = 10

        # Printing
        self.print_summary = False


class DDPGAgent(object):

    def __init__(self, params: DDPGAgentParams, example_state, example_action, stats=None):

        self.params = params
        self.gamma = tf.constant(self.params.gamma, dtype=float)
        self.align_counter = 0
        self.boolean_map_shape = example_state.get_boolean_map_shape()
        self.float_map_shape = example_state.get_float_map_shape()
        self.scalars = example_state.get_num_scalars(give_position=self.params.use_scalar_input)
        self.num_actions = len(example_action)
        #print('safsaffa',self.num_actions)
        self.num_map_channels = self.boolean_map_shape[2] + self.float_map_shape[2]
        #################################加了一些噪声因子##########################
        self.w_init = tf.random_normal_initializer(mean=0, stddev=0.3)
        self.b_init = tf.constant_initializer(0.1)
        self.var=self.params.var
        self.c_loss=0.0
        self.a_loss=0.0

        # Create shared inputs
        action_input = Input(shape=self.num_actions, name='action_input', dtype=tf.float32)
        reward_input = Input(shape=(), name='reward_input', dtype=tf.float32)
        termination_input = Input(shape=(), name='termination_input', dtype=tf.bool)
        q_star_input = Input(shape=(), name='q_star_input', dtype=tf.float32)

        if self.params.blind_agent:
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [scalars_input]
            self.q_network = self.build_blind_model(scalars_input)
            self.target_network = self.build_blind_model(scalars_input, 'target_')
            self.hard_update()

        elif self.params.use_scalar_input:
            devices_input = Input(shape=(3 * self.params.max_devices,), name='devices_input', dtype=tf.float32)
            uavs_input = Input(shape=(4 * self.params.max_uavs,), name='uavs_input', dtype=tf.float32)
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [devices_input,
                      uavs_input,
                      scalars_input]

            self.q_network = self.build_scalars_model(states)
            self.target_network = self.build_scalars_model(states, 'target_')
            self.hard_update()

        else:
            boolean_map_input = Input(shape=self.boolean_map_shape, name='boolean_map_input', dtype=tf.bool)
            float_map_input = Input(shape=self.float_map_shape, name='float_map_input', dtype=tf.float32)
            scalars_input = Input(shape=(self.scalars,), name='scalars_input', dtype=tf.float32)
            states = [boolean_map_input,
                      float_map_input,
                      scalars_input]

            self.a_network = self.build_actor_model(states, 'eval_a')
            self.targeta_network = self.build_actor_model(states, 'target_a')
            self.q_network = self.build_critic_model(states, action_input, 'eval_c')
            self.targetq_network = self.build_critic_model(states, action_input, 'target_c')
            self.hard_update()

            self.global_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                          outputs=self.global_map)
            self.local_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.local_map)
            self.total_map_model = Model(inputs=[boolean_map_input, float_map_input],
                                         outputs=self.total_map)

        # a_values=self.a_network.output
        # a_target_values=self.targeta_network.output
        # q_values = self.q_network.output
        # q_target_values = self.targetq_network.output
        # # Define the loss of a-network
        # # choose_action=self.a_network(states)
        # # Q_s_a=self.q_network(states,choose_action)
        # # a_loss = -tf.reduce_mean(Q_s_a)
        # # self.a_loss_model = Model(
        # #     inputs=states,
        # #     outputs=a_loss)
        #
        # # Define the loss of q-network.
        # # Exploit act model
        # #self.exploit_model = Model(inputs=states, outputs=max_action)
        # self.exploit_model_target = Model(inputs=states, outputs=a_target_values) #for testing
        #
        # # Softmax explore model
        # # softmax_scaling = tf.divide(q_values, tf.constant(self.params.soft_max_scaling, dtype=float))
        # # softmax_action = tf.math.softmax(softmax_scaling, name='softmax_action')
        # self.soft_explore_model = Model(inputs=states, outputs=a_values) #for training

        self.a_optimizer = tf.optimizers.Adam(learning_rate=params.actor_learning_rate, amsgrad=True)
        self.q_optimizer = tf.optimizers.Adam(learning_rate=params.critic_learning_rate, amsgrad=True)

        # if self.params.print_summary:
        #     self.a_loss_model.summary()
        #     self.q_loss_model.summary()

        if stats:
            stats.set_model(self.targetq_network)
            stats.set_model(self.targeta_network)
            # stats.add_log_data_callback('c_loss', self.get_c_loss())
            # stats.add_log_data_callback('a_loss', self.get_a_loss())


    def build_actor_model(self, inputs, name=''):
        boolean_map_input = inputs[0]
        float_map_input = inputs[1]
        scalars_input = inputs[2]
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
        padded_map = tf.concat([map_cast, float_map_input], axis=3)
        map_proc =padded_map
        states_proc=scalars_input
        flatten_map = self.create_map_proc(map_proc, name)
        layer = Concatenate(name=name + 'concat')([flatten_map, states_proc])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)
        model.summary()
        return model

    #########################################加一个critic的网络##########################
    def build_critic_model(self, states, actions,  name=''):
        boolean_map_input = states[0]
        float_map_input = states[1]
        scalars_input = states[2]
        map_cast = tf.cast(boolean_map_input, dtype=tf.float32)
        padded_map = tf.concat([map_cast, float_map_input], axis=3)
        map_proc = padded_map
        states_proc = scalars_input
        flatten_map = self.create_map_proc(map_proc, name)
        #####################3
        state_layer = Concatenate()([flatten_map, states_proc])
        layer = Concatenate(name=name + 'concat')([state_layer, actions])
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        model = Model(inputs=[states,actions], outputs=output)
        model.summary()
        return model

    #########################################加一个critic的网络##########################

    def build_scalars_model(self, inputs, name=''):

        layer = Concatenate(name=name + 'concat')(inputs)
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', kernel_initializer=self.w_init, bias_initializer=self.b_init, name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_blind_model(self, inputs, name=''):

        layer = inputs
        for k in range(self.params.hidden_layer_num):
            layer = Dense(self.params.hidden_layer_size, activation='relu', name=name + 'hidden_layer_all_' + str(k))(
                layer)
        output = Dense(self.num_actions, activation='linear', name=name + 'output_layer')(layer)
        model = Model(inputs=inputs, outputs=output)
        return model

    def create_map_proc(self, conv_in, name):

        # Forking for global and local map
        # Global Map
        global_map = tf.stop_gradient(
            AvgPool2D((self.params.global_map_scaling, self.params.global_map_scaling))(conv_in))

        self.global_map = global_map
        self.total_map = conv_in

        for k in range(self.params.conv_layers):
            global_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                                strides=(1, 1),
                                name=name + 'global_conv_' + str(k + 1))(global_map)

        flatten_global = Flatten(name=name + 'global_flatten')(global_map)

        # Local Map
        crop_frac = float(self.params.local_map_size) / float(self.boolean_map_shape[0])
        local_map = tf.stop_gradient(tf.image.central_crop(conv_in, crop_frac))
        self.local_map = local_map

        for k in range(self.params.conv_layers):
            local_map = Conv2D(self.params.conv_kernels, self.params.conv_kernel_size, activation='relu',
                               strides=(1, 1),
                               name=name + 'local_conv_' + str(k + 1))(local_map)

        flatten_local = Flatten(name=name + 'local_flatten')(local_map)

        return Concatenate(name=name + 'concat_flatten')([flatten_global, flatten_local])

    def act(self, state):
        #var=self.var
        a=self.get_soft_max_exploration(state)
        #print('the original action is :',a)
        #action_final = np.clip(np.random.normal(a, var), -1, 1)

        return a


    def get_random_action(self):
        output = np.random.uniform(-1, 1, size=self.num_actions)

        return output
    '''
    def get_exploitation_action(self, state):

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model(scalars).numpy()[0]

        if self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model([devices_in, uavs_in, scalars]).numpy()[0]

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.exploit_model([boolean_map_in, float_map_in, scalars]).numpy()[0]
    '''

    def get_soft_max_exploration(self, state):  ###############for training
        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            action = self.soft_explore_model(scalars).numpy()[0]
        elif self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[
                tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[
                tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            action = self.soft_explore_model([devices_in, uavs_in, scalars]).numpy()[0]
        else:
            boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
            float_map_in = state.get_float_map()[tf.newaxis, ...]
            scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

            action = self.a_network([boolean_map_in, float_map_in, scalars]).numpy()[0]

        return action

    def get_exploitation_action_target(self, state):  #################for test

        if self.params.blind_agent:
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]
            return self.exploit_model_target(scalars).numpy()[0]

        if self.params.use_scalar_input:
            devices_in = state.get_device_scalars(self.params.max_devices, relative=self.params.relative_scalars)[
                tf.newaxis, ...]
            uavs_in = state.get_uav_scalars(self.params.max_uavs, relative=self.params.relative_scalars)[
                tf.newaxis, ...]
            scalars = np.array(state.get_scalars(give_position=True), dtype=np.single)[tf.newaxis, ...]

            return self.exploit_model_target([devices_in, uavs_in, scalars]).numpy()[0]

        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        scalars = np.array(state.get_scalars(), dtype=np.single)[tf.newaxis, ...]

        return self.targeta_network([boolean_map_in, float_map_in, scalars]).numpy()[0]



    def hard_update(self):#更新参数，只用于首次赋值，之后就没用
        self.targeta_network.set_weights(self.a_network.get_weights())
        self.targetq_network.set_weights(self.q_network.get_weights())

    def soft_update(self, alpha):
        weights1 = self.q_network.get_weights()
        target_weights1 = self.targetq_network.get_weights()
        self.targetq_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights1, target_weights1)])
        weights2 = self.a_network.get_weights()
        target_weights2 = self.targeta_network.get_weights()
        self.targeta_network.set_weights(
            [w_new * alpha + w_old * (1. - alpha) for w_new, w_old in zip(weights2, target_weights2)])

    def train(self, experiences):
        boolean_map = experiences[0]
        float_map = experiences[1]
        scalars = tf.convert_to_tensor(experiences[2], dtype=tf.float32)
        action = tf.convert_to_tensor(experiences[3], dtype=tf.int64)
        reward = tf.convert_to_tensor(experiences[4], dtype=tf.float32)
        next_boolean_map = experiences[5]
        next_float_map = experiences[6]
        next_scalars = tf.convert_to_tensor(experiences[7], dtype=tf.float32)
        terminated = experiences[8]
        batch_states = [boolean_map, float_map, scalars]
        batch_states_ = [next_boolean_map, next_float_map, next_scalars]
        # Train Value network
        with tf.GradientTape() as tape:
            a_ = self.targeta_network(batch_states_)
            q_ = self.targetq_network([batch_states_, a_])
            gamma_terminated = tf.multiply(tf.cast(tf.math.logical_not(terminated), tf.float32), self.gamma)
            y = reward + gamma_terminated  * tf.squeeze(q_,axis=1)
            q = self.q_network([batch_states, action])
            # td_error = tf.losses.mean_squared_error(y, q)
            # import pdb
            # pdb.set_trace()
            critic_loss = tf.math.reduce_mean(tf.math.square(y - tf.squeeze(q,axis=1)))
            # print(td_error)
            # print(critic_loss)
            # self.c_loss=tf.reduce_mean(critic_loss)
        c_grads = tape.gradient(critic_loss, self.q_network.trainable_weights)
        self.q_optimizer.apply_gradients(zip(c_grads, self.q_network.trainable_weights))

        with tf.GradientTape() as tape:
            a=self.a_network(batch_states)
            q=self.q_network([batch_states,a])
            a_loss=-tf.reduce_mean(q)
            # import pdb
            # pdb.set_trace()
            self.a_loss = a_loss

        a_grads = tape.gradient(a_loss, self.a_network.trainable_weights)
        self.a_optimizer.apply_gradients(zip(a_grads, self.a_network.trainable_weights))
        self.soft_update(self.params.alpha)


    def save_weights(self, path_to_weights):
        self.target_network.save_weights(path_to_weights)

    def save_model(self, path_to_model):
        self.target_network.save(path_to_model)

    def load_weights(self, path_to_weights):
        self.q_network.load_weights(path_to_weights)
        self.a_network.load_weights(path_to_weights)
        self.hard_update()

    def get_global_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.global_map_model([boolean_map_in, float_map_in]).numpy()

    def get_local_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.local_map_model([boolean_map_in, float_map_in]).numpy()

    def get_total_map(self, state):
        boolean_map_in = state.get_boolean_map()[tf.newaxis, ...]
        float_map_in = state.get_float_map()[tf.newaxis, ...]
        return self.total_map_model([boolean_map_in, float_map_in]).numpy()
