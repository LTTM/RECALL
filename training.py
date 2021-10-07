import os
import data_loader
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import metrics
from PIL import Image
import shutil

from model_resnet import Model

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

class Training:

    def __init__(self, manager):
        self.manager = manager
        self.parameters = manager.get_parameters()
        self.batch_size = self.parameters["batch_size"]
        self.train_image_size = self.parameters["train_image_size"]
        self.interleave_ratio_old = self.parameters["interleave_ratio_old"]
        self.interleave_ratio_new = self.parameters["interleave_ratio_new"]
        self.training_steps_per_class = self.parameters["training_steps_per_class"]
        self.training_steps_per_class_incremental = self.parameters["training_steps_per_class_incremental"]
        self.training_steps_per_class_helper = self.parameters["training_steps_per_class_helper"]

        self.model = None
        self.classes = None
        self.old_classes = None
        self.new_classes = None

    def train(self):

        # 1) STEP 0 ENCODER + DECODER
        if not self.manager.check_encoder():
            name_encoder = self.manager.get_encoder_name()
            path_encoder = self.manager.get_encoder_path()
            name_decoder = self.manager.get_decoders_names()[0]
            path_decoder = self.manager.get_decoder_paths()[0]
            training_paths = self.manager.get_train_paths()[0]
            validation_paths = self.manager.get_val_paths()[0]
            validation_file_path = self.manager.get_validation_file_path()

            classes = [i for i in range(self.manager.get_encoder_number()+1)]

            print("TEST " + str(self.manager.get_test_index()) + " (" + self.manager.get_test_type() + ")\n" )
            print("STEP 0")
            print("ENCODER NAME: ", name_encoder)
            print("ENCODER PATH ", path_encoder)
            print("DECODER NAME: ", name_decoder)
            print("DECODER PATH ", path_decoder)
            print("CLASSES: ", classes)
            print("TRAINING: ", training_paths)
            print("VALIDATION: ", validation_paths)
            print("VALIDATE: ", self.manager.validate)
            print("VALIDATION_FILE: ", validation_file_path)

            
            self.train_first_step(classes=classes,
                                  train_path=training_paths,
                                  val_path=validation_paths,
                                  save_encoder=True,
                                  save_decoder=True,
                                  encoder_in_path=self.manager.get_encoder_pretrained_path(),
                                  encoder_out_path=path_encoder,
                                  decoder_out_path=path_decoder,
                                  validate=self.validate,
                                  validate_file_path=validation_file_path)

        # 2) HELPERS
        if not self.manager.check_helpers():
            print("STARTING TRAINING OF HELPER DECODERS...")
            helper_classes = self.manager.get_helpers_classes()
            helper_train_paths = self.manager.get_helper_train_paths()
            helper_out_paths = self.manager.get_helper_out_paths()

            for i in range(len(helper_classes)):
                classes = helper_classes[i]
                out_path = helper_out_paths[i]
                train_path = helper_train_paths[i]
                encoder_path = self.manager.get_encoder_path()
                print("TRAINING OF HELPER ", i, " :")
                print("OUT: ", out_path)
                print("TRAIN: ", train_path)
                print("CLASSES: ", classes)

                self.train_helper(helper_classes=classes,
                                  train_path=train_path,
                                  encoder_in_path=encoder_path,
                                  decoder_out_path=out_path)

    

        # 3) LABELS
        if not self.manager.check_labels_no_helper():

            # print("EVALUATING LABELS")
            # replay_tfrecord_paths = self.manager.get_replay_tfrecords_paths()

            encoder_classes = [i for i in range(self.manager.get_encoder_number()+1)]

            ################################# NO HELPER LABELS EVALUATION###############################################

            replay_classes = self.manager.get_replay_classes_no_helper()
            replay_outputs = self.manager.get_replay_output_no_helper_paths()
            replay_inputs  = self.manager.get_replay_source_no_helper_paths()
            replay_names   = self.manager.get_replay_no_helper_tfrecords_names()
            tmp_folder     = self.manager.get_replay_folder() + "/tmp/"
            debug_folder   = self.manager.get_replay_folder() + "/debug/"
            encoder_path = self.manager.get_encoder_path()
            decoder_path = self.manager.get_decoder_paths()[0]


            print("EVALUATING LABELS WITHOUT HELPER FOR CLASSES:")
            print(replay_classes)
            print("FROM: ", replay_inputs)
            print("TO: ", replay_outputs)

            for i in range(len(replay_classes)):
                current_input = replay_inputs[i]
                current_output = replay_outputs[i]
                current_debug_folder = debug_folder + replay_names[i]

                self.compute_labels(is_helper=False,
                                    helper_classes=None,
                                    encoder_classes=encoder_classes,
                                    encoder_path=encoder_path,
                                    decoder_path=decoder_path,
                                    source_tfrecord_path=current_input,
                                    tmp_path=tmp_folder,
                                    debug_path=current_debug_folder,
                                    out_tfrecord_path=current_output)

            shutil.rmtree(tmp_folder)

            ####################################### HELPERS LABELS EVALUATION ##########################################
        if not self.manager.check_labels_helper():

            replay_classes_all = self.manager.get_replay_classes_helper()
            replay_classes = self.manager.get_helpers_classes()
            replay_outputs = self.manager.get_replay_output_helper_paths()
            replay_inputs  = self.manager.get_replay_source_helper_paths()
            replay_names   = self.manager.get_replay_helper_tfrecords_names()
            tmp_folder     = self.manager.get_replay_folder() + "tmp/"
            debug_folder   = self.manager.get_replay_folder() + "debug/"
            encoder_path = self.manager.get_encoder_path()
            helper_paths = self.manager.get_helper_out_paths()


            print("EVALUATING LABELS WITH HELPER FOR CLASSES:")
            print(replay_classes_all)
            print("FROM: ", replay_inputs)
            print("TO: ", replay_outputs)

            for i in range(len(replay_classes)):

                current_classes = replay_classes[i]
                current_inputs = replay_inputs[i]
                current_outputs = replay_outputs[i]
                current_helper = helper_paths[i]
                current_names = replay_names[i]

                for j in range(len(current_classes)):

                    current_debug_folder = debug_folder + current_names[j]

                    print("current cl", current_classes)
                    print("input:", current_inputs[j])

                    self.compute_labels(is_helper=True,
                                        helper_classes=current_classes,
                                        encoder_classes=None,
                                        encoder_path=encoder_path,
                                        decoder_path=current_helper,
                                        source_tfrecord_path=current_inputs[j],
                                        tmp_path=tmp_folder,
                                        debug_path=current_debug_folder,
                                        out_tfrecord_path=current_outputs[j])

                shutil.rmtree(tmp_folder)

        # 4) INCREMENTAL STEPS
        if not self.manager.check_decoders():
            print("Starting incremental steps...")
            starting_step = self.manager.find_step()
            total_steps = self.manager.get_incremental_steps_n()
            assert starting_step<=total_steps, "Error, starting step > total steps"
            steps = [starting_step + i for i in range(total_steps-starting_step+1)]

            for s in steps:
                classes_old = self.manager.get_old_classes_for_incremental_step(s)
                classes_new = self.manager.get_new_classes_for_incremental_step(s)
                train_path_old = self.manager.get_replay_tfrecords_for_step(s)
                train_path_new = self.manager.get_train_paths()[s]
                val_path = self.manager.get_val_paths()[s]
                encoder_in_path = self.manager.get_encoder_path()
                decoder_in_path = self.manager.get_decoder_paths()[s-1]
                encoder_out_path = None
                decoder_out_path = self.manager.get_decoder_paths()[s]
                validate_file_path = self.manager.get_validation_file_path()

                print("train_path_old=", train_path_old)
                print("train_path_new= ", train_path_new)
                print("INCREMENTAL STEP ", s)
                print("OLD CLASSES: ", classes_old)
                print("NEW CLASSES: ", classes_new)





                ################################## MIXING LABELS IF NEEDED #############################################

                print("MIXING LABELS? ", self.manager.mix_labels())

                if self.manager.mix_labels():
                    tfrecord_out = self.manager.get_mixed_train_paths()[s]
                    print("TFRECORD OUT: ", tfrecord_out)
                    if not self.manager.exists_mixed_tfrecord(s):
                        tmp_folder = "data/training_incremental/mixed/tmp/"
                        self.mix_label(encoder_path=encoder_in_path,
                                       decoder_path=decoder_in_path,
                                       tfrecord_in=train_path_new,
                                       tfrecord_out=tfrecord_out,
                                       tmp_folder=tmp_folder,
                                       classes=classes_old)
                        shutil.rmtree(tmp_folder)

                    train_path_new = tfrecord_out
                ########################################################################################################


                self.train_incremental_step(step_n=s,
                                            classes_old=classes_old,
                                            classes_new=classes_new,
                                            train_path_old=train_path_old,
                                            train_path_new=train_path_new,
                                            val_path=val_path,
                                            save_encoder=False,
                                            save_decoder=True,
                                            encoder_in_path=encoder_in_path,
                                            decoder_in_path=decoder_in_path,
                                            encoder_out_path=encoder_out_path,
                                            decoder_out_path=decoder_out_path,
                                            validate=self.validate,
                                            validate_file_path=validate_file_path)

    def train_first_step(self,
                         classes,
                         train_path,
                         val_path,
                         save_encoder,
                         save_decoder,
                         encoder_in_path,
                         encoder_out_path,
                         decoder_out_path,
                         validate,
                         validate_file_path):

        self.model = Model(self.parameters)
        self.model.build(is_helper=False,
                         is_incremental=False,
                         classes=classes,
                         helper_classes=None,
                         old_classes=None,
                         new_classes=None)

        dataset = self._load_dataset_for_training([train_path])
        train_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        train_next = train_iter.get_next()
        train_input = train_next["input"]
        train_label = train_next["label"]

        with tf.Session(config=config) as sess:

            sess.run([tf.global_variables_initializer()])
            sess.run([tf.local_variables_initializer()])
            sess.run([train_iter.initializer])

            self._load_encoder(encoder_in_path, sess)

            training_steps = (len(classes)-1) * self.training_steps_per_class

            for i in tqdm(range(training_steps)):
                input_, label_ = sess.run([train_input, train_label])

                feed_dict = {self.model.inputs: input_, self.model.input_labels: label_}

                _ = sess.run(self.model.train_op, feed_dict=feed_dict)

            if save_decoder:
                self._save_decoder(decoder_out_path, sess)
            if save_encoder:
                self._save_encoder(encoder_out_path, sess)
            if validate:
                self.validate(sess,
                              validation_path=val_path,
                              old_classes=None,
                              new_classes=classes,
                              validation_step=0,
                              output_file=validate_file_path)

        self.model = None
        tf.keras.backend.clear_session()


    ####### VALIDATE FIRST STEP

    def eval_first_step(self,
                         classes,
                         val_path,
                         encoder_path,
                         decoder_path,
                         validate_file_path):

        self.model = Model(self.parameters)
        self.model.build(is_helper=False,
                         is_incremental=False,
                         classes=classes,
                         helper_classes=None,
                         old_classes=None,
                         new_classes=None)

        with tf.Session(config=config) as sess:

            sess.run([tf.global_variables_initializer()])
            sess.run([tf.local_variables_initializer()])
    
            self._load_encoder(encoder_path, sess)
            self._load_decoder(decoder_path, sess)
            
            self.validate(sess,
                          validation_path=val_path,
                          old_classes=None,
                          new_classes=classes,
                          validation_step=0,
                          output_file=validate_file_path)

        self.model = None
        tf.keras.backend.clear_session()




    def train_incremental_step(self,
                               step_n,
                               classes_old,
                               classes_new,
                               train_path_old,
                               train_path_new,
                               val_path,
                               save_encoder,
                               save_decoder,
                               encoder_in_path,
                               decoder_in_path,
                               encoder_out_path,
                               decoder_out_path,
                               validate,
                               validate_file_path):

        self.model = Model(self.parameters)
        self.model.build(is_helper=False,
                         is_incremental=True,
                         classes=classes_old + classes_new,
                         helper_classes=None,
                         old_classes=classes_old,
                         new_classes=classes_new)

        buffer_size = 400 * (len(classes_old)-1)

        dataset = self._load_interleaved_datasets(train_path_old,
                                                  [train_path_new],
                                                  repeat=True,
                                                  shuffle=True,
                                                  buffer_size=buffer_size)

        train_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        train_next = train_iter.get_next()
        train_input = train_next["input"]
        train_label = train_next["label"]

        with tf.Session(config=config) as sess:

            sess.run([tf.global_variables_initializer()])
            sess.run([tf.local_variables_initializer()])
            sess.run([train_iter.initializer])

            self._load_encoder(encoder_in_path, sess)
            self._load_decoder(decoder_in_path, sess)

            training_steps = len(classes_new) * self.training_steps_per_class_incremental

            print("FILLING BUFFERS FOR SHUFFLING...")
            for i in tqdm(range(training_steps)):

                input_, label_ = sess.run([train_input, train_label])
                feed_dict = {self.model.inputs: input_, self.model.input_labels: label_}

                _ = sess.run(self.model.train_op_decoder, feed_dict=feed_dict)

            if save_decoder:
                self._save_decoder(decoder_out_path, sess)
            if save_encoder:
                self._save_encoder(encoder_out_path, sess)
            if validate:
                self.validate(sess,
                              validation_path=val_path,
                              old_classes=classes_old,
                              new_classes=classes_new,
                              validation_step=step_n,
                              output_file=validate_file_path)

        self.model = None
        tf.keras.backend.clear_session()



    def train_helper(self,
                     helper_classes,
                     train_path,
                     encoder_in_path,
                     decoder_out_path
                     ):

        self.model = Model(self.parameters)
        self.model.build(is_helper=True,
                         is_incremental=False,
                         classes=None,
                         helper_classes=helper_classes,
                         old_classes=None,
                         new_classes=None)

        dataset = self._load_dataset_for_training([train_path])
        train_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        train_next = train_iter.get_next()
        train_input = train_next["input"]
        train_label = train_next["label"]

        with tf.Session(config=config) as sess:

            sess.run([tf.global_variables_initializer()])
            sess.run([tf.local_variables_initializer()])
            sess.run([train_iter.initializer])

            self._load_encoder(encoder_in_path, sess)

            training_steps = len(helper_classes) * self.training_steps_per_class_helper

            for i in tqdm(range(training_steps)):
                input_, label_ = sess.run([train_input, train_label])

                feed_dict = {self.model.inputs: input_, self.model.input_labels: label_}

                _ = sess.run(self.model.train_op_decoder, feed_dict=feed_dict)


            self._save_decoder(decoder_out_path, sess)

        self.model = None
        tf.keras.backend.clear_session()

    def compute_labels(self,
                       is_helper=False,
                       helper_classes=None,
                       encoder_classes=None,
                       encoder_path=None,
                       decoder_path=None,
                       source_tfrecord_path=None,
                       tmp_path=None,
                       debug_path=None,
                       out_tfrecord_path=None):

            images_folder = tmp_path + "/images/"
            labels_folder = tmp_path + "/labels/"
            rgb_labels_folder = tmp_path + "/rgb_labels/"

            images_folder_debug = debug_path + "/images/"
            labels_folder_debug = debug_path + "/labels/"
            rgb_labels_folder_debug = debug_path + "/rgb_labels/"

            if not os.path.exists(images_folder):
                os.makedirs(images_folder)
            if not os.path.exists(labels_folder):
                os.makedirs(labels_folder)
            if not os.path.exists(rgb_labels_folder):
                os.makedirs(rgb_labels_folder)
            if not os.path.exists(images_folder_debug):
                os.makedirs(images_folder_debug)
            if not os.path.exists(labels_folder_debug):
                os.makedirs(labels_folder_debug)
            if not os.path.exists(rgb_labels_folder_debug):
                os.makedirs(rgb_labels_folder_debug)

            self.model = Model(self.parameters)

            if is_helper:
                self.model.build(is_helper=True,
                                 is_incremental=False,
                                 classes=None,
                                 helper_classes=helper_classes,
                                 old_classes=None,
                                 new_classes=None)
            else:
                self.model.build(is_helper=False,
                                 is_incremental=False,
                                 classes=encoder_classes,
                                 helper_classes=None,
                                 old_classes=None,
                                 new_classes=None)


            with tf.Session(config=config) as sess:

                # DATASET AND ITERATOR
                replay_dataset = self._load_dataset_only_images(source_tfrecord_path)
                replay_iter = replay_dataset.make_initializable_iterator()
                iterations = self._estimate_iterations(sess, replay_iter)
                next = replay_iter.get_next()
                input = next["input"]

                # INITIALIZERS
                sess.run([replay_iter.initializer])
                sess.run([tf.global_variables_initializer()])
                sess.run([tf.local_variables_initializer()])

                # LOAD VARIABLES
                self._load_encoder(encoder_path, sess)
                self._load_decoder(decoder_path, sess)


                for i in tqdm(range(iterations)):
                    input_ = sess.run(input)
                    feed_dict = {self.model.inputs: input_}

                    image, label, visible = sess.run([self.model.inputs,
                                                      self.model.outputs_argmax,
                                                      self.model.outputs_rgb],
                                                      feed_dict=feed_dict)

                    # IMAGES
                    image = Image.fromarray(np.array(image).squeeze().astype(np.uint8))
                    label = Image.fromarray(np.array(label).squeeze().astype(np.uint8))
                    visible = Image.fromarray(np.array(visible).squeeze())

                    # SAVE
                    image.save(images_folder + str(i) + ".jpg")
                    label.save(labels_folder + str(i) + ".png")
                    visible.save(rgb_labels_folder + str(i) + ".jpg")

                    # SAVE DEBUG SAMPLES
                    if i % 20 == 0:
                        image.save(images_folder_debug + str(i) + ".jpg")
                        label.save(labels_folder_debug + str(i) + ".png")
                        visible.save(rgb_labels_folder_debug + str(i) + ".jpg")

                # CREATE TF RECORD
                data_loader.tfrecord_from_folder(folders=[images_folder, labels_folder], output=out_tfrecord_path,
                                                include_labels=True)
            self.model = None
            tf.keras.backend.clear_session()

        
########################################################################################################################
    def mix_label(self, encoder_path, decoder_path, tfrecord_in, tfrecord_out, tmp_folder, classes):

        self.model = Model(self.parameters)

        self.model.build(is_helper=False,
                         is_incremental=False,
                         classes=classes,
                         helper_classes=None,
                         old_classes=None,
                         new_classes=None)

        # MAKING FOLDERS
        images_folder = tmp_folder + "/images/"
        labels_folder = tmp_folder + "/labels/"

        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
            os.makedirs(images_folder)
            os.makedirs(labels_folder)

        with tf.Session(config=config) as sess:
            # DATASET AND ITERATOR

            dataset = data_loader.load_tfrecord([tfrecord_in], mode="validation+").batch(1)
            iter = dataset.make_initializable_iterator()
            iterations = self._estimate_iterations(sess, iter)
            next = iter.get_next()
            input = next["input"]
            label = next["label"]
            rgb_input = next["rgb_input"]

            # INITIALIZERS
            sess.run([iter.initializer])
            sess.run([tf.global_variables_initializer()])
            sess.run([tf.local_variables_initializer()])

            # LOAD VARIABLES
            self._load_encoder(encoder_path, sess)
            self._load_decoder(decoder_path, sess)


            for i in tqdm(range(iterations)):
                input_, label_, rgb_input_ = sess.run([input, label, rgb_input])
                feed_dict = {self.model.inputs: input_,
                             self.model.labels: label_}


                label_n, output_n = sess.run([self.model.labels,
                                              self.model.outputs_argmax],
                                              feed_dict=feed_dict)

                # compute mixed labels
                input_image = np.array(rgb_input_).squeeze().astype(np.uint8)
                input_label = np.array(label_n).squeeze().astype(np.uint8)
                input_label[input_label == 255] = 0
                output_argmax = np.array(output_n).squeeze().astype(np.uint8)
                label_mixed = self.mix_labels(label_on=input_label, label_under=output_argmax)

                # convert to images
                input_image = Image.fromarray(input_image)
                label_mixed = Image.fromarray(label_mixed)

                # save images
                input_image.save(images_folder + str(i).zfill(4) + ".jpg")
                label_mixed.save(labels_folder + str(i).zfill(4) + ".png")


            data_loader.tfrecord_from_folder(folders=[images_folder, labels_folder], output=tfrecord_out,
                                            include_labels=True)
        self.model = None
        tf.keras.backend.clear_session()


    def mix_labels(self, label_on, label_under):
        """
           Mix two labels. The nonzero pixels of the first label are all kept, while the zero pixels of it are
           ovewritten by the corresponding pixels of the second label.

           :param label_on: The first label.
           :param label_under. The second label.

           :return: the mixed label as described above.

        """
        condition = label_on != 0
        label_under[condition] = 0
        return label_on + label_under


    # NOT CURRENTLY USED FUNCTION
    #
    def save_outputs(self,
                     encoder_path,
                     decoder_path,
                     validation_paths,
                     output_folder):

        self.model = Model(self.parameters)

        self.model.build(is_helper=False,
                         is_incremental=False,
                         classes=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                         helper_classes=None,
                         old_classes=None,
                         new_classes=None)

        with tf.Session(config=config) as sess:

            for i, file in enumerate(validation_paths):

                current_path = output_folder + "/" + str(i+1) + "/"
                images_path = current_path + "images/"
                outputs_path = current_path + "outputs/"
                labels_path = current_path + "labels/"

                if not os.path.exists(current_path):
                    os.makedirs(current_path)
                if not os.path.exists(images_path):
                    os.makedirs(images_path)
                if not os.path.exists(outputs_path):
                    os.makedirs(outputs_path)
                if not os.path.exists(labels_path):
                    os.makedirs(labels_path)

                print(file)
                # DATASET AND ITERATOR
                dataset = data_loader.load_tfrecord([file], mode="validation+").batch(1)
                iter = dataset.make_initializable_iterator()
                iterations = self._estimate_iterations(sess, iter)
                next = iter.get_next()
                input = next["input"]
                label = next["label"]
                rgb_input = next["rgb_input"]

                # INITIALIZERS
                sess.run([iter.initializer])
                sess.run([tf.global_variables_initializer()])
                sess.run([tf.local_variables_initializer()])

                # LOAD VARIABLES
                self._load_encoder(encoder_path, sess)
                self._load_decoder(decoder_path, sess)

                # validation loop
                for j in tqdm(range(iterations)):

                    input_, label_, in_rgb = sess.run([input, label, rgb_input])

                    feed_dict = {self.model.inputs: input_,
                                 self.model.labels: label_}

                    out_rgb, lab_rgb = sess.run([self.model.outputs_rgb, self.model.labels_rgb], feed_dict=feed_dict)

                    # compute mixed labels
                    in_rgb = np.array(in_rgb).squeeze().astype(np.uint8)
                    lab_rgb = np.array(lab_rgb).squeeze().astype(np.uint8)
                    out_rgb = np.array(out_rgb).squeeze().astype(np.uint8)

                    # convert to images
                    in_rgb = Image.fromarray(in_rgb)
                    lab_rgb = Image.fromarray(lab_rgb)
                    out_rgb = Image.fromarray(out_rgb)

                    # save images
                    in_rgb.save(images_path + str(j).zfill(4) + ".jpg")
                    lab_rgb.save(labels_path + str(j).zfill(4) + ".png")
                    out_rgb.save(outputs_path + str(j).zfill(4) + ".png")

        self.model = None
        tf.keras.backend.clear_session()


    # validate the current model

    def validate(self,
                 sess,
                 validation_path,
                 old_classes,
                 new_classes,
                 validation_step,
                 output_file):

        print("Starting validation of step: ", validation_step)
        print("Dataset used: ", validation_path)
        print("Output file: ", output_file)

        dataset = self._load_dataset_for_validation(validation_path)
        val_iter = tf.compat.v1.data.make_initializable_iterator(dataset)
        iterations = self._estimate_iterations(sess, val_iter)

        next = val_iter.get_next()
        input = next["input"]
        label = next["label"]

        sess.run(val_iter.initializer)

        if old_classes is None:
            classes = new_classes
        else:
            classes = old_classes + new_classes

        cm = None
        losses = None

        # validation loop
        for i in tqdm(range(iterations)):

            input_, label_ = sess.run([input, label])

            feed_dict = {self.model.inputs: input_, self.model.labels: label_}

            loss_, cm_ = sess.run([self.model.loss, self.model.confusion_matrix], feed_dict=feed_dict)

            if i == 0:
                cm = cm_
                losses = [loss_]
            else:
                cm = cm + cm_
                losses.append(loss_)

        # IoU
        class_iou = metrics.get_classes_iou(cm, classes)
        miou = metrics.get_mean_iou(cm, classes)

        # Accuracy
        class_acc = metrics.get_classes_accuracy(cm, classes)
        macc = metrics.get_mean_class_accuracy(cm, classes)
        macc_pixel = metrics.get_mean_pixel_accuracy(cm, classes)

        # Metrics old classes vs new classes
        if old_classes is not None and len(old_classes) > 0:
            old_miou, new_miou = metrics.get_mean_iou_old_and_new(cm, old_classes, new_classes)
            old_acc, new_acc = metrics.get_mean_accuracy_old_and_new(cm, old_classes, new_classes)

        # Format
        for x in range(len(class_iou)):
            if class_iou[x] is not None:
                class_iou[x] = '{:.3f}'.format(class_iou[x])

        for x in range(len(class_acc)):
            if class_acc[x] is not None:
                class_acc[x] = '{:.3f}'.format(class_acc[x])

        with open(output_file, "a+") as f:

            print("\n################################# VALIDATION RESULTS OF STEP " +
            str(validation_step) + "####################################", file=f)
            print("LOSS\n" + str(sum(losses) / len(losses)) + "\n", file=f)
            print("mIoU\n" + str(miou) + "\n", file=f)
            print("class_iou\n" + str(class_iou) + "\n", file=f)
            print("mean class accuracy\n" + str(macc) + "\n", file=f)
            print("mean pixel accuracy\n" + str(macc_pixel) + "\n", file=f)
            print("class accuracy\n" + str(class_acc) + "\n", file=f)

            if old_classes is not None and len(old_classes) > 0:
                print("old classes miou " + str(old_classes) + "\n" + str(old_miou) + "\n", file=f)
                print("new classes miou " + str(new_classes) + "\n" + str(new_miou) + "\n", file=f)
                print("old classes mean accuracy " + str(old_classes) + "\n" + str(old_acc) + "\n", file=f)
                print("new classes mean accuracy " + str(new_classes) + "\n" + str(new_acc) + "\n", file=f)


#############################################################################
    def validate_ckpt(self,step_n):
        
        # encoder/decoder
        encoder_path = self.manager.get_encoder_path()
        decoder_path = self.manager.get_decoder_paths()[step_n]
        # validation tfrecords
        validation_paths = self.manager.get_val_paths()[step_n]
        # classes
        classes_old = self.manager.get_old_classes_for_incremental_step(step_n)
        classes_new = self.manager.get_new_classes_for_incremental_step(step_n)
        #out file
        output_file = self.manager.get_validation_file_path()
       


        self.model = Model(self.parameters)
        self.model.build(is_helper=False,
                         is_incremental=True,
                         classes=classes_old + classes_new,
                         helper_classes=None,
                         old_classes=classes_old,
                         new_classes=classes_new)

        with tf.Session(config=config) as sess:

            sess.run([tf.global_variables_initializer()])
            sess.run([tf.local_variables_initializer()])

            self._load_encoder(encoder_path, sess)
            self._load_decoder(decoder_path, sess)

            print(validation_paths)
            self.validate(
                 sess,
                 validation_path=validation_paths,
                 old_classes=classes_old,
                 new_classes=classes_new,
                 validation_step=step_n,
                 output_file=output_file)

        self.model = None
        tf.keras.backend.clear_session()

############################################################################

    # FUnction to estimate the total number of iterations

    def _estimate_iterations(self, sess, iter):

        sess.run([iter.initializer])
        next = iter.get_next()

        iterations = 0
        while True:
            try:
                sess.run(next)
                iterations += 1

            except tf.errors.OutOfRangeError:
                break

        return iterations

    # Functions to save and restore encoder/decoder weigths

    def _load_encoder(self, path, session):
        assert self.model is not None, "self.model is None, can't load encoder!"
        if self.model.load_variables(session, self.model.get_encoder_variables(), path):
            print("Encoder from [" + path + "] loaded!")
        else:
            print("Error while loading encoder from [" + path + "]")

    def _load_decoder(self, path, session):
        assert self.model is not None, "self.model is None, can't load decoder!"

        if self.model.load_variables(session, self.model.get_decoder_variables(), path):
            print("Decoder from [" + path + "] loaded!")
        else:
            print("Error while loading decoder from [" + path + "]")

    def _save_encoder(self, path, session):
        assert self.model is not None, "self.model is None, can't save encoder!"

        if self.model.save_variables(session, self.model.get_encoder_variables(), path):
            print("Encoder saved to [" + path + "]")
        else:
            print("Error while saving encoder to [" + path + "]")

    def _save_decoder(self, path, session):
        assert self.model is not None, "self.model is None, can't save decoder!"

        if self.model.save_variables(session, self.model.get_decoder_variables(), path):
            print("Decoder saved to [" + path + "]")
        else:
            print("Error while saving decoder to [" + path + "]")


    # Functions to load one or more tfrecords into a dataset.

    def _load_dataset_only_images(self, path):
        return data_loader.load_tfrecord([path], mode="images").batch(1)

    def _load_dataset_for_training(self, path):
        return data_loader.load_tfrecord(path, mode="training",
                                        train_image_size=self.train_image_size). \
            batch(self.batch_size).shuffle(1000).repeat()

    def _load_dataset_for_validation(self, path):
        return data_loader.load_tfrecord([path], mode="validation").batch(1)

    def _load_interleaved_datasets(self, files_old, files_new, repeat=True, shuffle=True, buffer_size=100):

        dataset_new = data_loader.load_tfrecord(files_new, mode="training",
                                               train_image_size=self.train_image_size).batch(1)

        dataset_old = data_loader.load_tfrecord(files_old, mode="training",
                                               train_image_size=self.train_image_size).batch(1)

        return data_loader.interleave_datasets(dataset_new,
                                               dataset_old,
                                               self.interleave_ratio_new,
                                               self.interleave_ratio_old,
                                               self.batch_size,
                                               buffer_size=buffer_size,
                                               repeat=repeat,
                                               shuffle=shuffle)
