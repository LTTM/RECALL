import tensorflow as tf
import operations_resnet as ops
import six
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DEBUG = True


class Model(object):

    def __init__(self, parameters):

        # read_hyperparameters
        self.helper_size = parameters["helper_size"]
        self.n_classes = parameters["n_classes"]
        self.dataset_mean = parameters["dataset_mean"]
        self.helper_decoders_one_class = parameters["helper_decoder_one_class"]
        self.momentum = parameters["momentum"]
        self.start_decay_step = parameters["start_decay_step"]
        self.decay_steps_per_class = parameters["decay_steps_per_class"]
        self.decay_steps_per_class_incremental = parameters["decay_steps_per_class_incremental"]
        self.decay_steps_per_class_helper = parameters["decay_steps_per_class_helper"]
        self.decay_power = parameters["decay_power"]
        self.l2_decay = parameters["l2_decay"]
        self.lr_initial_value = parameters["lr_initial_value"]
        self.lr_final_value = parameters["lr_final_value"]
        self.lr_initial_value_incremental = parameters["lr_initial_value_incremental"]
        self.lr_final_value_incremental = parameters["lr_final_value_incremental"]
        self.lr_initial_value_helper = parameters["lr_initial_value_helper"]
        self.lr_final_value_helper = parameters["lr_final_value_helper"]

        # global step
        self.global_step = None

        # classes to be considered
        self.classes = None
        self.new_classes = None
        self.old_classes = None

        # mode of the model: incremental or helper?
        self.is_incremental = None
        self.is_helper = None

        # axis of the channels
        self.channel_axis = 3

        # lists of variables
        self.trainable_variables = None
        self.encoder_variables = None
        self.decoder_variables = None
        self.weights_variables = None
        self.encoder_weights_variables = None
        self.decoder_weights_variables = None

        # inputs of the network
        self.inputs = None
        self.input_labels = None
        self.labels = None
        self.labels_rgb = None
        self.one_hot_labels = None

        # outputs of the network
        self.output_shape = None
        self.features = None
        self.outputs = None
        self.outputs_argmax = None
        self.outputs_rgb = None

        # losses
        self.loss = None
        self.L2_loss_encoder = None
        self.L2_loss_decoder = None

        # optimizer and learning rate
        self.optimizer_name = "optimizer"
        self.optimizer = None
        self.learning_rate = None

        # training operations
        self.train_op = None
        self.train_op_encoder = None
        self.train_op_decoder = None

        # metrics for evaluation
        self.confusion_matrix = None

    def build(self, is_helper, is_incremental, classes=None, helper_classes=None, old_classes=None, new_classes=None):
        """ Main function to build the graph of the model and to generate the training operations.
        There are different hyperparameters for the training of the first step, of the incremental steps and of the
        helper decoders. The parameters of this function specify which is the considered case.
        If is_helper and is_incremental are both False we are training a model in its first step. If just is_helper is
        we are training an helper decoder and if just is_incremental is True we are training the model in an
        incremental step. is_helper and is_incremental cannot be True both at the same time.

        -First step training: is_helper FALSE and is_incremental FALSE. The hyperparameters for first step training
        are used. classes should be set to a list of the classes to be considered in the training of this step.
        -Incremental step training: is_helper FALSE and is_incremental TRUE. The hyperparameters for incremental step
        training are used. classes, old_classes and new_classes should be specified.
        -Helper training. The hyperparameters for the helper training are used. helper_classes should be specified.

        :param is_helper: boolean. True if we are training an helper decoder.
        :param is_incremental: boolean. True if we are training the model in an incremental step.
        :param classes: classes to be considere in first step training and in incremental steps.
        :param helper_classes: the classes to be considered in the training of the helper decoder.
        :param old_classes: already learnt classes for the training of the incremental step.
        :param new_classes: the new classes to add in the current incremental step.

        """

        # train of helper decoder
        if is_helper:
            assert helper_classes is not None, "Helper classes should be set!"
            assert not is_incremental, "Both is_helper and is_incremental are True!"
            self.is_helper = True
            self.is_incremental = False
            self.classes = [i for i in range(len(helper_classes)+1)]
            self.n_classes = len(helper_classes)
        else: # case: train of first step or incremental step
            assert classes is not None, "Classes should be set!"
            self.is_helper = False
            self.is_incremental = is_incremental
            self.classes = classes

        # In case we are in an incremental step
        if is_incremental:
            assert old_classes is not None and new_classes is not None, "New and Old classes should be set!"
            self.old_classes = old_classes
            self.new_classes = new_classes
            self.classes = old_classes + new_classes

        # Set the global step
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

        # input images and labels
        self.inputs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None, 3))
        self.input_labels = tf.compat.v1.placeholder(tf.uint8, shape=(None, None, None, 1))

        if is_helper:       # Train of an helper decoder

            # labels based on the classes of the corrent helper
            self.labels = self._get_helper_label(helper_classes)  # labels considering just the helper classes
            self.labels_rgb = Model._label2rgb(self.labels)       # labels RGB
            self.one_hot_labels = Model._label2onehot(self.labels, len(helper_classes)+1)

            # encoded features and decoded ourput
            self.output_shape = tf.shape(self.inputs)
            self.features = self.encode(self.inputs)
            self.outputs = self.decode(self.features, len(helper_classes)+1)

            # argmax output and rgb output
            self.outputs_argmax = self._get_helper_argmax_output(helper_classes)
            self.outputs_rgb = Model._output2rgb(self.outputs_argmax, False)

        else:       # Train of the main model

            # labels
            self.labels = self.input_labels
            self.labels_rgb = Model._label2rgb(self.labels)
            self.one_hot_labels = Model._label2onehot(self.labels, self.n_classes)

            # encoded features and decoded output
            self.output_shape = tf.shape(self.inputs)
            self.features = self.encode(self.inputs)
            self.outputs = self.decode(self.features, self.n_classes)

            # argmax output and rgb output
            self.outputs_argmax = tf.argmax(self.outputs, axis=3)
            self.outputs_rgb = Model._output2rgb(self.outputs_argmax, False)

        # VARIABLES LISTS
        self._set_variable_lists()

        # Losses
        self.loss = self._get_loss(self.classes)
        #self.loss = self._get_loss([0]+new_classes)
        self.L2_loss_encoder = self._get_L2_loss_encoder()
        self.L2_loss_decoder = self._get_L2_loss_decoder()

        # Confusion matrix
        self.confusion_matrix = self._get_confusion_matrix()

        # Learning Rate
        self.learning_rate = self._get_learning_rate()

        # Optimizer
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    momentum=self.momentum,
                                                    name=self.optimizer_name)

        # Update lists of variables
        self._set_variable_lists()

        # Training operations
        self.train_op = self.optimizer.minimize(tf.reduce_sum(self.loss + self.L2_loss_encoder + self.L2_loss_decoder),
                                                global_step=self.global_step,
                                                var_list=self.encoder_variables + self.decoder_variables)

        self.train_op_encoder = self.optimizer.minimize(self.loss + self.L2_loss_encoder,
                                                        global_step=self.global_step,
                                                        var_list=self.encoder_variables)

        self.train_op_decoder = self.optimizer.minimize(self.loss + self.L2_loss_decoder,
                                                        global_step=self.global_step,
                                                        var_list=self.decoder_variables)


    def _get_helper_label(self, classes):
        """ Computes the labels for the training of the helper decoder
        :param classes: the list of classes of the helper decoder.
        :return: the labels for helper decoder training
        """
        labels_helper = tf.zeros_like(self.input_labels, dtype=tf.uint8)

        for i in enumerate(classes):
            current = tf.math.scalar_mul(i[0]+1, tf.cast(tf.math.equal(self.input_labels, i[1]), tf.uint8))
            labels_helper = tf.math.add(labels_helper, current)

        return labels_helper


    def _get_helper_argmax_output(self, classes):
        """ Computes the argmax output of the helper decoder.
            :param classes: the list of classes of the helper decoder.
            :return: the argmax output.
        """
        classes = [0] + classes
        output = tf.argmax(self.outputs, axis=3)
        table = tf.constant(classes, tf.int32)
        output = tf.nn.embedding_lookup(table, output)
        return output

    def _set_variable_lists(self):
        """ Update the variable lists, keeping track of all variable associated to the encoder or to the decoder."""

        global_variables = tf.global_variables()
        self.encoder_variables = [v for v in global_variables if 'resnet_v1_101' in v.name]
        self.decoder_variables = [v for v in global_variables if 'decoder' in v.name]
        self.weights_variables = [v for v in global_variables if 'weights' in v.name]
        self.encoder_weights_variables = [v for v in self.weights_variables if 'resnet_v1_101' in v.name]
        self.decoder_weights_variables = [v for v in self.weights_variables if 'decoder' in v.name]


    def _get_loss(self, classes):
        """ Compute the loss considering just a list of classes.
            :param classes: the classes to be considered.
            :return: the loss
        """

        # Compute loss
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.outputs, labels=self.one_hot_labels)

        mask = tf.math.greater(tf.reduce_sum(tf.map_fn(lambda b: tf.cast(tf.math.equal(self.labels, b), tf.uint8),
                                                       tf.constant(classes, dtype=tf.uint8)), axis=0), 0)
        mask = tf.reshape(mask, tf.shape(self.labels))

        mask = tf.squeeze(mask, axis=[-1])

        # apply mask
        loss = tf.boolean_mask(loss, mask)

        # mean of the values
        loss = tf.reduce_mean(loss)

        return loss

    
    def _get_L2_loss_encoder(self):
        """ Compute the L2 loss of the encoder.
            :return: the L2 loss of the encoder.
        """

        l2_loss = [self.l2_decay * tf.nn.l2_loss(v) for v in self.encoder_weights_variables]
        return tf.add_n(l2_loss)


    def _get_L2_loss_decoder(self):
        """ Compute the L2 loss of the decoder.
            :return: the L2 loss of the decoder.
        """
        l2_loss = [self.l2_decay * tf.nn.l2_loss(v) for v in self.decoder_weights_variables]
        return tf.add_n(l2_loss)

    # Learning rate


    def _get_learning_rate(self):
        """ Computes a decayed learning rate based on some hyperparameters and on the current training step
            (global_step). The hyperparameters may differ if we are training the model in the first step, in an
            incremental step or if we are training an helper decoder.

            :return: the learning rate.
        """

        if self.is_incremental:
            decay_steps = len(self.new_classes) * self.decay_steps_per_class_incremental
            lr_final_value = self.lr_final_value_incremental
            lr_initial_value = self.lr_initial_value_incremental
        elif self.is_helper:
            decay_steps = len(self.classes) * self.decay_steps_per_class_helper
            lr_final_value = self.lr_final_value_helper
            lr_initial_value = self.lr_initial_value_helper
        else:
            decay_steps = len(self.classes) * self.decay_steps_per_class
            lr_final_value = self.lr_final_value
            lr_initial_value = self.lr_initial_value

        step = tf.maximum(self.global_step - self.start_decay_step, 0)
        step = tf.minimum(step, decay_steps)

        decayed_learning_rate = (lr_initial_value - lr_final_value) * \
                                (1 - step / decay_steps) ** self.decay_power + self.lr_final_value

        return decayed_learning_rate


    # Confusion matrix
    def _get_confusion_matrix(self):
        """ Computes the confusion matrix.
            :return: the confusion matrix.
        """
        # create the mask
        mask = tf.math.greater(tf.reduce_sum(tf.map_fn(lambda b: tf.cast(tf.math.equal(self.labels, b), tf.uint8),
                                                       tf.constant(self.classes, dtype=tf.uint8)), axis=0), 0)
        mask = tf.reshape(mask, tf.shape(self.labels))

        mask = tf.squeeze(mask, axis=[-1])

        output_masked = tf.boolean_mask(self.outputs_argmax, mask)
        label_masked = tf.boolean_mask(self.labels, mask)

        # output_masked = tf.reshape(output_masked, [-1,])
        label_masked = tf.dtypes.cast(label_masked, tf.int64)

        return tf.confusion_matrix(predictions=output_masked, labels=label_masked, num_classes=self.n_classes)

    # Util functions

    @staticmethod
    def _output2rgb(image, do_argmax=True):
        """
        Convert the output tensor of the generator to an RGB image
        :param image: output tensor given by the generator. 4D tensor: [batch_size, image_width, image_height, num_classes]
        :return: image converted to RGB format. 4D tensor: [batch_size, image_width, image_height, 3]
        """
        table = tf.constant(
            [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
             [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
             [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
             [128, 192, 0], [0, 64, 128]], tf.uint8)

        if do_argmax:
            image = tf.argmax(image, axis=3)

        out_RGB = tf.nn.embedding_lookup(table, image)
        return out_RGB

    @staticmethod
    def _label2rgb(image):
        """
        Convert the tensor containing the GT to an RGB image
        :param image: tensor containing gt labels. 4D tensor: [batch_size, image_width, image_height, 1]
        :return: image converted to RGB format. 4D tensor: [batch_size, image_width, image_height, 3]
        """

        table = tf.constant(
            [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128], [128, 0, 128], [0, 128, 128],
             [128, 128, 128], [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
             [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0], [128, 64, 0], [0, 192, 0],
             [128, 192, 0], [0, 64, 128],
             # fix wrong classes using the value of the background (Only for better visualization of gt)
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0],
             [0, 0, 0], [0, 0, 0], [255, 255, 255]], tf.uint8)

        labels = tf.squeeze(image, axis=3)
        labels = tf.cast(labels, tf.int32)
        out_RGB = tf.nn.embedding_lookup(table, labels)
        return out_RGB


    @staticmethod
    def _label2onehot(label, num_classes):
        """
        This function one-hot encodes the label
        :param label: 4D tensor of dimensions [batch_size, image_width, image_height, 1]
        :return: 4D tensor of dimensions [batch_size, image_width, image_height, num_classes]
        """

        # Prep the data. Make sure the labels are in one-hot format
        if len(label.get_shape()) == 4:
            label = tf.squeeze(label, axis=3)

        label = tf.cast(label, tf.uint8)
        label = tf.one_hot(label, num_classes)

        return label

    # variable lists functions

    # get all encoder variables
    def get_encoder_variables(self):
        return self.encoder_variables

    # get all decoder variables
    def get_decoder_variables(self):
        return self.decoder_variables

    # get all weights variables
    def get_weights_variables(self):
        return self.weights_variables

    # get encoder weights variables
    def get_encoder_weights_variables(self):
        return self.encoder_weights_variables

    # get decoder weights variables
    def get_decoder_weights_variables(self):
        return self.decoder_weights_variables

    # Save and Load variables from/to file

    def save_variables(self, session, variables, path, global_step=None):

        if DEBUG:
            print("Saving weights to ", path)

        try:
            saver = tf.compat.v1.train.Saver(variables)
            saver.save(session, path, global_step, write_meta_graph=False, write_state=False)

            if DEBUG:
                print("Weights saved correctly!")
            return True
        except:
            if DEBUG:
                print("Error while saving weights!")
            return False

    def load_variables(self, session, variables, path):

        if DEBUG:
            print("Restoring weights from ", path)

        try:
            loader = tf.compat.v1.train.Saver(variables, reshape=True)
            loader.restore(session, path)

            if DEBUG:
                print("Weights restored correctly!")
            return True
        except:
            if DEBUG:
                print("Error while loading weights!")
            return False

    ########################################### MODEL ##################################################################

    def encode(self, input):

        with tf.compat.v1.variable_scope('resnet_v1_101', reuse=tf.AUTO_REUSE):

            if DEBUG:
                print("-----------build encoder: deeplab pre-trained-----------")
                print("decoder input:", input.shape)

            outputs = self._start_block(input, 'conv1')

            if DEBUG:
                print("after start block:", outputs.shape)

            with tf.variable_scope('block1') as scope:
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_1', identity_connection=False)
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_2')
                outputs = self._bottleneck_resblock(outputs, 256, 'unit_3')

            if DEBUG:
                print("after block1:", outputs.shape)

            with tf.variable_scope('block2') as scope:
                outputs = self._bottleneck_resblock(outputs, 512, 'unit_1', half_size=True, identity_connection=False)

                for i in six.moves.range(2, 5):
                    outputs = self._bottleneck_resblock(outputs, 512, 'unit_%d' % i)

            if DEBUG:
                print("after block2:", outputs.shape)

            with tf.variable_scope('block3') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_1', identity_connection=False)

                for i in six.moves.range(2, 24):
                    outputs = self._dilated_bottle_resblock(outputs, 1024, 2, 'unit_%d' % i)

            if DEBUG:
                print("after block3:", outputs.shape)

            with tf.variable_scope('block4') as scope:
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_1', identity_connection=False)
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_2')
                outputs = self._dilated_bottle_resblock(outputs, 2048, 4, 'unit_3')

            if DEBUG:
                print("encoder output:", outputs.shape)

            return outputs

    def decode(self, input, n_classes):
        with tf.compat.v1.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            if DEBUG:
                print("-----------build decoder-----------")

            outputs = ops.ASPP(input, n_classes, [6, 12, 18, 24], channel_axis=self.channel_axis)

            if DEBUG:
                print("after ASPP block:", outputs.shape)

            outputs = ops.upsampling(outputs, self.output_shape)

            if DEBUG:
                print("Output shape:", outputs.shape)

            return outputs

    ####################################### BLOCKS ####################################################################
    def _start_block(self, input, name):
        outputs = ops.conv2d(input, 7, 64, 2, name=name, channel_axis=self.channel_axis)
        outputs = ops.batch_norm(outputs, name=name, is_training=False, activation_fn=tf.nn.relu)
        outputs = ops.max_pool2d(outputs, 3, 2, name='pool1')

        return outputs

    def _bottleneck_resblock(self, x, num_o, name, half_size=False, identity_connection=True):
        first_s = 2 if half_size else 1
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = ops.conv2d(x, 1, num_o, first_s,  name='%s/bottleneck_v1/shortcut' % name, channel_axis=self.channel_axis)
            o_b1 = ops.batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = ops.conv2d(x, 1, num_o / 4, first_s, name='%s/bottleneck_v1/conv1' % name, channel_axis=self.channel_axis)
        o_b2a = ops.batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = ops.conv2d(o_b2a, 3, num_o / 4, 1, name='%s/bottleneck_v1/conv2' % name, channel_axis=self.channel_axis)
        o_b2b = ops.batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = ops.conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name, channel_axis=self.channel_axis)
        o_b2c = ops.batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = ops.add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = ops.relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs

    def _dilated_bottle_resblock(self, x, num_o, dilation_factor, name, identity_connection=True):
        assert num_o % 4 == 0, 'Bottleneck number of output ERROR!'
        # branch1
        if not identity_connection:
            o_b1 = ops.conv2d(x, 1, num_o, 1, name='%s/bottleneck_v1/shortcut' % name, channel_axis=self.channel_axis)
            o_b1 = ops.batch_norm(o_b1, name='%s/bottleneck_v1/shortcut' % name, is_training=False, activation_fn=None)
        else:
            o_b1 = x
        # branch2
        o_b2a = ops.conv2d(x, 1, num_o / 4, 1, name='%s/bottleneck_v1/conv1' % name, channel_axis=self.channel_axis)
        o_b2a = ops.batch_norm(o_b2a, name='%s/bottleneck_v1/conv1' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2b = ops.dilated_conv2d(o_b2a, 3, num_o / 4, dilation_factor, name='%s/bottleneck_v1/conv2' % name,
                                   channel_axis=self.channel_axis)
        o_b2b = ops.batch_norm(o_b2b, name='%s/bottleneck_v1/conv2' % name, is_training=False, activation_fn=tf.nn.relu)

        o_b2c = ops.conv2d(o_b2b, 1, num_o, 1, name='%s/bottleneck_v1/conv3' % name, channel_axis=self.channel_axis)
        o_b2c = ops.batch_norm(o_b2c, name='%s/bottleneck_v1/conv3' % name, is_training=False, activation_fn=None)
        # add
        outputs = ops.add([o_b1, o_b2c], name='%s/bottleneck_v1/add' % name)
        # relu
        outputs = ops.relu(outputs, name='%s/bottleneck_v1/relu' % name)
        return outputs
