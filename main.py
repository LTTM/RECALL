import os
from train_manager import TrainManager
from training import Training
import tensorflow as tf
import gin


def main(unused_args):

    TEST_TYPES = ["ADD_ONE_ONE_TIME",
                  "ADD_FIVE_ONE_TIME",
                  "ADD_FIVE_TWO_TIMES",
                  "ADD_ONE_FIVE_TIMES",
                  "ADD_TEN_ONE_TIME",
                  "ADD_TWO_FIVE_TIMES",
                  "ADD_ONE_TEN_TIMES"]

    REPLAY_SOURCES = ["gan", "flickr"]
    CONFIGS = ['OVERLAPPED', 'DISJOINT', 'SEQUENTIAL']  

    # joint training
    print('******************************RUNNING EXPERIMENT: JOINT TRAINING')
    gin.parse_config_file('config/JOINT.gin')
    manager = TrainManager(test_type='JOINT', replay_source=None)
    training = Training(manager)
    training.train()

    # incremental training
    for r in REPLAY_SOURCES:
        for conf in CONFIGS:
            for i in TEST_TYPES:
                tf.keras.backend.clear_session()

                print(f'******************************RUNNING EXPERIMENT: {i} {conf} {r}')

                gin.parse_config_file('config/' + conf + '.gin')
                manager = TrainManager(test_type=i, replay_source=r)
                training = Training(manager)
                training.train()



if __name__ == '__main__':
    tf.compat.v1.app.run()
