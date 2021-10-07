import os
from pathlib import Path
#from distutils.dir_util import copy_tree
#from shutil import copyfile
import gin

@gin.configurable
class TrainManager:

    # available incremental tests

    TEST_0 = ("ADD_ONE_ONE_TIME", [19, 1])
    TEST_1 = ("ADD_FIVE_ONE_TIME", [15, 5])
    TEST_2 = ("ADD_FIVE_TWO_TIMES", [10, 5, 5])
    TEST_3 = ("ADD_ONE_FIVE_TIMES", [15, 1, 1, 1, 1, 1])
    TEST_4 = ("ADD_TEN_ONE_TIME", [10, 10])
    TEST_5 = ("ADD_TWO_FIVE_TIMES", [10, 2, 2, 2, 2, 2])
    TEST_6 = ("ADD_ONE_TEN_TIMES", [10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    TEST_7 = ("JOINT", [20])

    TESTS = [TEST_0, TEST_1, TEST_2, TEST_3, TEST_4, TEST_5, TEST_6, TEST_7]


    def __init__(self,

                 # PARAMETERS OF CLASS 0: TEST PARAMETERS
                 test_name,
                 test_type,
                 helper_decoders_one_class,
                 replay_source,
                 base_folder_path,
                 validate,

                 # PARAMETERS OF CLASS 1: COMMON PARAMETERS
                 parameters_common_index,
                 batch_size,
                 n_classes,
                 dataset_mean,
                 train_image_size,
                 momentum,
                 start_decay_step,
                 decay_power,
                 l2_decay,

                 # PARAMETERS OF CLASS 2: FIRST STEP PARAMETERS
                 parameters_fs_index,
                 dataset_type_first_step,
                 lr_initial_value,
                 lr_final_value,
                 training_steps_per_class,
                 decay_steps_per_class,

                 # PARAMETERS OF CLASS 3: HELPER PARAMETERS
                 parameters_helper_index,
                 dataset_type_helpers,
                 lr_initial_value_helper,
                 lr_final_value_helper,
                 training_steps_per_class_helper,
                 decay_steps_per_class_helper,

                 # PARAMETERS OF CLASS 4: INCREMENTAL PARAMETERS
                 parameters_incremental_index,
                 dataset_type_incremental,
                 mix_labels,
                 lr_initial_value_incremental,
                 lr_final_value_incremental,
                 training_steps_per_class_incremental,
                 decay_steps_per_class_incremental,
                 interleave_ratio_old,
                 interleave_ratio_new
                 ):

        """
        :param test_name: String. The name associated to the current incremental test or setup.
        :param test_type: String. Which sequence of classes to consider among the available:
                -"ADD_ONE_ONE_TIME" = test 19-1
                -"ADD_FIVE_ONE_TIME" = test 15-5
                -"ADD_FIVE_TWO_TIMES" = test 10-5
                -"ADD_ONE_FIVE_TIMES" = test 15-1
                -"ADD_TEN_ONE_TIME" = test 10-10
                -"ADD_TWO_FIVE_TIMES" = test 10-2
                -"ADD_ONE_TEN_TIMES" = test 10-1
                -"JOINT" = joint training of all 20 classes.
        :param helper_decoders_one_class: True to use one-class helpers even if incremental steps use more
        classes.
        :param replay_source: String "gan" or "flickr" to select the replay source.
        :param base_folder_path: base folder of the tree.
        :param validate: True if the validation process should occur after the training.

        :param parameters_common_index: Integer. The index of used common parameters.
        :param batch_size: The batch size to be used.
        :param dataset_mean: The dataset mean to subtract to input images.
        :param train_image_size: The size of images during training.
        :param momentum: Momentum for the momentum optimized.
        :param start_decay_step: Starting step for the decay.
        :param decay_power: the decay power.
        :param l2_decay: the decay for l2 regularization.

        :param parameters_fs_index: Integer. Index of used first step parameters.
        :param dataset_type_first_step: String. Can be specified to be "sequential" or "overlapped" in order to use one
        of the two setups for the first step. The setup "disjoint" is equal to "seauential" in the first step of
        training so it is not needed here.
        :param lr_initial_value: The initial value for the learning rate in the first-step training.
        :param lr_final_value: The final value for the learning rate in the first-step training.
        :param training_steps_per_class: the number of training steps, for each class, for the first-step
        training.
        :param decay_steps_per_class: the number of the dacay steps for the learning rate, for each class, in the first-
        step training.

        :param parameters_helper_index: Integer. Index of used helper parameters.
        :param dataset_type_helpers: String. Can be specified to be "sequential", "disjoint" or "overlapped" in order to
        use one of the different dataset for the training of the helper decoders.
        :param lr_initial_value_helper: The initial value for the learning rate in the helper training.
        :param lr_final_value_helper: The final value for the learning rate in the helper training.
        :param training_steps_per_class_helper: the number of training steps, for each class, for the helper decoders
        training.
        :param decay_steps_per_class_helper: the number of the dacay steps for the learning rate, for each class, for
        the training of helper decoders

        :param parameters_incremental_index: Integer. Index of used parameters for incremental steps.
        :param dataset_type_incremental: String. Can be specified to be "sequential", "disjoint" or "overlapped" in
         order to use one of the different dataset for the incremental training.
        :param mix_labels: Boolean. If True the labels of the dataset are mixed with the output of the current model,
        before proceeding with the training of the current incremental step.
        :param lr_initial_value_incremental: The initial value for the learning rate in the incremental step training.
        :param lr_final_value_incremental: The final value for the learning rate in the incremental step training.
        :param training_steps_per_class_incremental: the number of training steps, for each class, for the training of
        the incremental steps.
        :param decay_steps_per_class_helper: the number of the dacay steps for the learning rate, for each class, for
        the training of the incremental steps.
        :param interleave_ratio_old: ratio of old generated samples to be interleved into the training dataset.
        :param interleave_ratio_new: ratio of new samples to be interleaved into the training dataset.
        """

        # basic info
        self.test_name = test_name
        self.test_type = test_type
        self.base_folder_path = base_folder_path

        # parameters indices
        self.parameters_common_index = parameters_common_index
        self.parameters_fs_index = parameters_fs_index
        self.parameters_helper_index = parameters_helper_index
        self.parameters_incremental_index = parameters_incremental_index

        # dataset types
        self.dataset_type_first_step = dataset_type_first_step
        self.dataset_type_helpers = dataset_type_helpers
        self.dataset_type_incremental = dataset_type_incremental
        self.mix_labels_incremental = mix_labels
        self.helper_decoders_one_class = helper_decoders_one_class

        self.replay_source = replay_source
        assert replay_source is None or replay_source == "gan" or replay_source == "flickr", \
        "Replay source not set correctly"

        # validate?
        self.validate = validate

        # derived parameters
        self.test_structure, self.test_index = TrainManager._get_test_structure_and_index(self.test_type)

        assert self.test_structure is not None, "test structure is None!"

        self.parameters = {
            "batch_size": batch_size,
            "helper_size": self.get_helper_size(),
            "n_classes": n_classes,
            "dataset_mean": dataset_mean,
            "train_image_size": train_image_size,
            "helper_decoder_one_class": helper_decoders_one_class,
            "momentum": momentum,
            "start_decay_step": start_decay_step,
            "training_steps_per_class": training_steps_per_class,
            "decay_steps_per_class": decay_steps_per_class,
            "training_steps_per_class_incremental": training_steps_per_class_incremental,
            "decay_steps_per_class_incremental": decay_steps_per_class_incremental,
            "training_steps_per_class_helper": training_steps_per_class_helper,
            "decay_steps_per_class_helper": decay_steps_per_class_helper,
            "decay_power": decay_power,
            "l2_decay": l2_decay,
            "lr_initial_value": lr_initial_value,
            "lr_final_value": lr_final_value,
            "lr_initial_value_incremental": lr_initial_value_incremental,
            "lr_final_value_incremental": lr_final_value_incremental,
            "lr_initial_value_helper": lr_initial_value_helper,
            "lr_final_value_helper": lr_final_value_helper,
            "interleave_ratio_old": interleave_ratio_old,
            "interleave_ratio_new": interleave_ratio_new}


    # do we need to mix labels?
    def mix_labels(self):

        return (self.dataset_type_incremental == "disjoint" or self.dataset_type_incremental == "overlapped") \
               and self.mix_labels_incremental


    # GET PATH functions


    def get_encoder_pretrained_path(self):
        """
        Return the path of the checkpoint of the pretrained encoder.
        :return: Path of the pretrained encoder.
        """
        return self.base_folder_path + "/checkpoints/encoders/pretrained"


    def get_encoder_path(self):
        """
        Return the path of the checkpoint of the step-0 encoder for the current test.
        :return: Path of the step-0 encoder.
        """
        return self.base_folder_path + "/checkpoints/encoders/" + self.get_encoder_name()


    def get_decoder_paths(self):
        """
        Return the paths of the checkpoints of all decoders for the current step.
        :return: A list of strings. Each string is a decoder path for the current test.
        """
        return [self.base_folder_path + "/checkpoints/decoders/" + n for n in self.get_decoders_names()]


    def get_helper_out_paths(self):
        """
        Return the paths of the checkpoints of all helper decoders for the current step.
        :return: A list of strings. Each string is an helper decoder path for the current test.
        """
        helper_folder_path = self.base_folder_path + "/checkpoints/helper_decoders/"

        return [helper_folder_path + n for n in self.get_helper_names()]


    def get_validation_file_path(self):
        """
        Return the path of the validation txt file.
        :return: Path of the validation txt file.
        """
        validation_file_name = self.get_validation_file_name()
        if self.helper_decoders_one_class:
            validation_file_name = validation_file_name + "_1"

        return self.base_folder_path + "/outputs/" + validation_file_name + ".txt"


    def get_train_paths(self):
        """
        Returns the training file paths for every step of the current test.
        :return: A list of lists of strings. Every String is path to a tfrecord training file. Every inner list contains
        the training files for a training step. The training step index is the index of the inner list.
        """

        assert self.dataset_type_first_step == "sequential" or self.dataset_type_first_step == "overlapped", \
            "Error! dataset_type_first_step should be set to 'sequential' or to 'overlapped'"

        assert self.dataset_type_incremental == "sequential" or self.dataset_type_incremental == "overlapped" or \
               self.dataset_type_incremental == "disjoint", \
            "Error! dataset_type_incremental should be set to 'sequential', to 'overlapped' or to 'disjoint'"

        steps = self.test_structure_to_steps()
        paths = []

        for s in enumerate(steps):
            if s[0] == 0:
                folder = "training_first_step"
                suffix = self.dataset_type_first_step
            else:
                folder = "training_incremental"
                suffix = self.dataset_type_incremental

            if s[1][0] == s[1][-1]:
                class_id = str(s[1][0])
            else:
                class_id = str(s[1][0]) + "-" + str(s[1][-1])

            current_path = self.base_folder_path + "/data/" + folder + "/train_" + suffix + \
                               "_" + class_id + ".tfrecord"

            paths.append(current_path)

        return paths


    def get_mixed_train_paths(self):
        assert self.dataset_type_first_step == "sequential" or self.dataset_type_first_step == "overlapped", \
            "Error! dataset_type_first_step should be set to 'sequential' or to 'overlapped'"

        assert self.dataset_type_incremental == "sequential" or self.dataset_type_incremental == "overlapped" or \
               self.dataset_type_incremental == "disjoint", \
            "Error! dataset_type_incremental should be set to 'sequential', to 'overlapped' or to 'disjoint'"

        steps = self.test_structure_to_steps()
        paths = []

        for s in enumerate(steps):
            if s[0] == 0:
                paths.append(None)
            else:
                suffix = self.dataset_type_incremental
                if s[1][0] == s[1][-1]:
                    class_id = str(s[1][0])
                else:
                    class_id = str(s[1][0]) + "-" + str(s[1][-1])

                current_path = self.base_folder_path + "/data/training_incremental/mixed/train_" + suffix + \
                           "_" + class_id + "_P" + self.parameter_to_str() + ".tfrecord"

                paths.append(current_path)

        return paths


    def get_helper_train_paths(self):
        paths = []
        helper_classes = self.get_helpers_classes()
        for h in helper_classes:
            path = self.base_folder_path + "/data/training_incremental/"

            if h[0] == h[-1]:
                class_id = str(h[0])
            else:
                class_id = str(h[0]) + "-" + str(h[-1])

            path = path + "train_" + self.dataset_type_helpers + "_" + class_id + ".tfrecord"
            paths.append(path)

        return paths


    def get_val_paths(self):
        """
        Returns the validation file paths for every step of the current test.
        :return: A list of lists of strings. Every String is path to a tfrecord validation file. Every inner list
        contains the validation files for a validation step. The validation step index is the index of the inner list.
        """
        steps = self.test_structure_to_steps()
        paths = []

        for s in steps:
            current_paths = []
            last_class = s[-1]
            for i in range(last_class):
                path = self.base_folder_path + "/data/validation/val_" + str(i + 1) + ".tfrecord"
                current_paths.append(path)

            paths.append(current_paths)

        return paths


    def get_replay_source_helper_paths(self):
        """
        Returns all tfrecord file paths of images that need an helper to compute their labels.
        :return: A list of lists of strings. Each inner list is associated to an helper (with the same index).
        Each string is a path to a tfrecord file containing images. This file need an
        helper for the computation of labels.
        """

        if self.replay_source is None:
            return None

        paths = []
        classes = self.get_helpers_classes()

        base_path = self.base_folder_path + "/data/replay_images/" + self.replay_source + "/"
        for hc in classes:
            current_paths = []
            for c in hc:
                path = base_path + str(c).zfill(2) + ".tfrecord"
                current_paths.append(path)
            paths.append(current_paths)
        return paths


    def get_replay_source_no_helper_paths(self):
        """
        Returns all tfrecord file paths of images that do not need an helper to compute their labels.
        :return: A list of strings. Each string is a path to a tfrecord file containing images. This file do not need an
        helper for the computation of labels.
        """
        paths = []
        classes = self.get_replay_classes_no_helper()
        base_path = self.base_folder_path + "/data/replay_images/" + self.replay_source + "/"

        
        for c in classes:
            full_path = base_path + str(c).zfill(2) + ".tfrecord"
            paths.append(full_path)
        
        return paths


    def get_replay_output_helper_paths(self):
        paths = []
        names = self.get_replay_helper_tfrecords_names()
        for hc in names:
            current_paths = []
            for n in hc:
                path = self.get_replay_folder() + n + ".tfrecord"
                current_paths.append(path)
            paths.append(current_paths)
        return paths


    def get_replay_output_no_helper_paths(self):
        names = self.get_replay_no_helper_tfrecords_names()

        return [self.get_replay_folder() + n + ".tfrecord" for n in names]


    def get_replay_output_paths(self):
        replay_no_helper = self.get_replay_output_no_helper_paths()
        replay_helper = self.get_replay_output_helper_paths()
        # flat list
        replay_helper = [item for sublist in replay_helper for item in sublist]

        return replay_no_helper + replay_helper


    # GET FOLDER function


    def get_replay_folder(self):
        """
        Returns the output folder for tfrecords of replay images and labels
        :return: A string containing the path.
        """

        images_with_labels_folder = self.base_folder_path + "/data/replay_images_and_labels/"
        return images_with_labels_folder

 
    # GET NAME functions


    def get_encoder_name(self):
        """
        Returns the name (without extension) of the encoder of the current test.
        :return: A string containing the name (without extension) of the encoder of the current test.
        """
        return "P" + str(self.parameters_common_index) + "." + str(self.parameters_fs_index) + "_E" \
               + str(self.get_encoder_number())


    def get_decoders_names(self):
        """
        Returns the names (without extension) of the decoders of the current test.
        :return: A list of strings. Every string contains the name (without extension) of a decoder for the current
        test.
        """
        if self.replay_source is None:
            return ["P" + str(self.parameters_common_index) + "." + str(self.parameters_fs_index) + "_E" \
                            + str(self.get_encoder_number())]

        if self.helper_decoders_one_class:
            decoders_names = ["P" + str(self.parameters_common_index) + "." +
                              str(self.parameters_fs_index) + "." +
                              str(self.parameters_helper_index) + "." +
                              str(self.parameters_incremental_index) +
                              "_T" + str(self.test_index) + "_S" + str(i) + "_" +
                              self.replay_source + "_1" for i in range(len(self.test_structure))]
        else:
            decoders_names = ["P" + str(self.parameters_common_index) + "." +
                              str(self.parameters_fs_index) + "." +
                              str(self.parameters_helper_index) + "." +
                              str(self.parameters_incremental_index) +
                              "_T" + str(self.test_index) + "_S" + str(i) + "_" +
                              self.replay_source for i in range(len(self.test_structure))]

        decoders_names[0] = "P" + str(self.parameters_common_index) + "." + str(self.parameters_fs_index) + "_E" \
                            + str(self.get_encoder_number())

        return decoders_names


    def get_helper_names(self):
        """
        Returns the names (without extension) of the helper decoders of the current test.
        :return: A list of strings. Every string contains the name (without extension) of an helper decoder for the
        current test.
        """
        names = []

        if self.replay_source is None or self.test_type == 'JOINT':
            return names

        for c in self.get_helpers_classes():
            helper_name = "P" + str(self.parameters_common_index) + "." + \
                          str(self.parameters_fs_index) + "." + \
                          str(self.parameters_helper_index) + "_E" + \
                          str(self.get_encoder_number()) + "_C" + str(c[0]) + \
                          "-" + str(c[-1])

            names.append(helper_name)

        return names


    def get_replay_helper_tfrecords_names(self):
        """
        Returns the names (without extension) of the replay tfrecords for the current step.
        :return: A list of strings. Every string contains the name (without extension) of a replay tfrecord for the
        current test.
        """
        names = []

        if self.replay_source is None or self.test_type == 'JOINT':
            return names

        
        helper_classes = self.get_helpers_classes()

        for hc in helper_classes:
            current_names = []
            for c in hc:
                name = "P" + str(self.parameters_common_index) + "." + \
                       str(self.parameters_fs_index) + "." + \
                       str(self.parameters_helper_index) + \
                       "_E" + str(self.get_encoder_number()) + \
                       "_H" + str(self.get_helper_size()) + \
                       "_C" + str(c) + "_" + self.replay_source
                current_names.append(name)
            names.append(current_names)

        return names


    def get_replay_no_helper_tfrecords_names(self):
        """
        """
        names = []
        classes_no_helper = self.get_replay_classes_no_helper()

        if self.replay_source is None or self.test_type == 'JOINT':
            return names
        
        for c in classes_no_helper:
            name = "P" + str(self.parameters_common_index) + "." + \
                     str(self.parameters_fs_index) + "." + \
                    str(self.parameters_helper_index) + \
                    "_E" + str(self.get_encoder_number()) + \
                    "_H0_C" + str(c) + "_" + self.replay_source
            names.append(name)

        return names


    def parameter_to_str(self):
        return str(self.parameters_common_index) + "." + str(self.parameters_fs_index) + "." + \
               str(self.parameters_helper_index) + "." + str(self.parameters_incremental_index)


    def get_validation_file_name(self):
        """
         Returns the name (without extension) of the validation file
         :return: A String. The name of the validation output file.
         """
        name = self.test_name + " (T" + str(self.test_index) + "_P" + str(self.parameters_common_index) + "." + \
               str(self.parameters_fs_index) + "." + \
               str(self.parameters_helper_index) + "." + \
               str(self.parameters_incremental_index)

        if self.replay_source is not None:
            name = name + "_"+ self.replay_source

        if self.helper_decoders_one_class:
            name = name + "_1"

        name = name + ")"

        return name


    # GET TEST INFO functions


    def get_test_type(self):
        """
        The type of the current test.
        :return: A string describing the current test.
        """
        return self.test_type


    def get_test_index(self):
        """
        The index of the current test.
        :return: An integer, corresponding to the index of the current test.
        """
        return self.test_index


    def get_validate(self):
        """
        Validate attribute. True if every step of the test need a validation.
        :return: A boolean specifying if every step of the test need to be validate.
        """
        return self.validate


    def get_test_structure(self):
        """
         The structure of the current test is returned.
         :return: A list of integers specifying the structure of the current test.
         """
        return self.test_structure


    def get_parameters(self):
        """
         The parameters of the current test.
         :return: A dictionary containing the parameters of the current test
         """
        return self.parameters


    def get_encoder_number(self):
        """
        The number of classes the encoder has been trained.
        :return: An integer. The number of classes the encoder has been trained.
        """
        return TrainManager.TESTS[self.test_index][1][0]


    def get_helper_size(self):
        """
        An helper is a decoder trained just on a limited number of classes (i.e. the size of the helper).
        Find the size of helpers needed for the current test.
        :return: 0 if no helper is needed, the number of the helper otherwise.
        """

        incremental_structure = self.test_structure[1:]

        for i in incremental_structure:
            assert i == incremental_structure[0], "Not all incremental steps have the same amount of classes!"

        if len(incremental_structure) <= 1:
            return 0

        # if we have more then one incremental step but we set to use one-class helpers
        if self.helper_decoders_one_class:
            return 1

        # otherwise we use an helper that cover all classes of one incremental step
        return incremental_structure[0]


    # GET CLASSES functions

    def get_replay_classes_no_helper(self):
        """
        This function returns all the classes that need a replay in the current test but don't need an helper to be
        replayed.
        :return: A list of integers. Every interger is a class that need to be replayed but dosn't need an helper.
        """
        if self.replay_source is None or self.test_type == 'JOINT':
            return []

        return [i + 1 for i in range(self.get_encoder_number())]


    def get_replay_classes_helper(self):
        """
        This function returns all the classes that need a replay in the current test and they need an helper decoder.
        :return: A list of integers. Every interger is a class that need an helper decoder for the replay. An empty
        list if no class need an helper.
        """

        if self.replay_source is None or self.test_type == 'JOINT':
            return []

        class_from = self.get_encoder_number() + 1

        class_to = sum(self.test_structure) - self.test_structure[-1]

        if class_from > class_to:
            return []

        return [class_from + i for i in range((class_to - class_from) + 1)]


    def get_helpers_classes(self):
        """
        An helper is a decoder trained just on a limited number of classes (i.e. the size of the helper).
        Find all the helpers needed for the current test.
        :return: A list of lists. Any inner list contains the classes of that helper. An empty list if no helper is
        needed
        """

        helpers = []
        size = self.get_helper_size()

        if self.replay_source is None or self.test_type == 'JOINT':
            return helpers

        # helpers are not needed
        if size == 0:
            return helpers

        classes = self.get_replay_classes_helper()

        assert len(classes) % size == 0, "Classes to help are not divisible by the the helper number!"

        number_of_helpers = len(classes) // size

        for i in range(number_of_helpers):
            helpers.append(classes[i * size:(i + 1) * size])

        return helpers


    @staticmethod
    def _get_test_structure_and_index(type):

        for i in range(len(TrainManager.TESTS)):
            if type == TrainManager.TESTS[i][0]:
                return TrainManager.TESTS[i][1], i
        return None


    # CHECK functions

    def check_encoder(self):
        """
        Check if the encoder for the current test is available.
        :return: True if the encoder is available. False otherwise.
        """
        file = Path(self.get_encoder_path() + ".data-00000-of-00001")

        return file.exists()


    def check_decoders(self):
        """
        Check if all decoders are available.
        :return: True if all decoders are available. False otherwise.
        """

        for d in self.get_decoder_paths():
            full_path = d + ".data-00000-of-00001"
            file = Path(full_path)
            if not file.exists():
                return False

        return True


    def check_helpers(self):
        """
        An helper is a decoder trained just on a limited number of classes (i.e. the number of the helper).
        Check if the helpers for the current step are available.
        :return: True if all helpers are available. False otherwise.
        """
        paths = self.get_helper_out_paths()

        for p in paths:
            full_path = p + ".data-00000-of-00001"
            file = Path(full_path)
            if not file.exists():
                return False

        return True


    def get_old_classes_for_incremental_step(self, step):
        assert step <= self.get_incremental_steps_n() and step >= 1, "Error in step number!"
        n_classes = sum(self.get_test_structure()[0:step])
        return [i for i in range(n_classes + 1)]


    def get_new_classes_for_incremental_step(self, step):
        assert step <= self.get_incremental_steps_n() and step >= 1, "Error in step number!"
        step_classes_n = self.get_test_structure()[step]
        first_class_in_step = sum(self.get_test_structure()[0:step]) + 1
        return [first_class_in_step + i for i in range(step_classes_n)]


    def get_incremental_steps_n(self):
        return len(self.get_test_structure()) - 1


    def get_replay_tfrecords_for_step(self, step):
        assert step <= self.get_incremental_steps_n() and step >= 1, "Error in step number!"
        n_classes = sum(self.get_test_structure()[0:step])
        return self.get_replay_output_paths()[0:n_classes]


    def check_labels_no_helper(self):
        paths = self.get_replay_output_no_helper_paths()
        for p in paths:
            file = Path(p)
            if not file.exists():
                return False
        return True


    def check_labels_helper(self):
        paths = self.get_replay_output_helper_paths()

        # flat the list
        paths = [item for sublist in paths for item in sublist]

        for p in paths:
            file = Path(p)
            if not file.exists():
                return False
        return True


    def find_step(self):
        """
        Find the current incremental step.
        :return: the index of the incremental step. -1 if the Test is finished.
        """
        for p in enumerate(self.get_decoder_paths()):
            full_path = p[1] + ".data-00000-of-00001"
            file = Path(full_path)
            if not file.exists():
                return p[0]

        return -1


    def test_structure_to_steps(self):
        """
        Convert a test structure into a list of lists. Every inner list contains the classes for that step
        :return: A list of lists.
        """
        test_structure = self.test_structure
        classes = []
        counter = 1
        for c in test_structure:
            current = []
            for i in range(c):
                current.append(counter)
                counter += 1
            classes.append(current)
        return classes


    def exists_mixed_tfrecord(self, step):
        """
        Test if the mixed tfrecord exists for the current specified step.

        :param step: the current step to verify.
        :return: A boolean. True if the mixed tfrecord exists, False otherwise.
        """
        if step == 0:
            return False

        paths = self.get_mixed_train_paths()
        return os.path.exists(paths[step])

