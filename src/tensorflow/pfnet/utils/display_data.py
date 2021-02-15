#!/usr/bin/env python3

import matplotlib.pyplot as plt
import datautils, arguments

def display_data(params):
    """
    display data with the parsed arguments
    """

    # evaluation data
    test_ds = datautils.get_dataflow(params.testfiles, params.batch_size, is_training=False)

    itr = test_ds.as_numpy_iterator()
    raw_record = next(itr)
    data_sample = datautils.transform_raw_record(raw_record, params)

    true_states = data_sample['true_states']
    global_map = data_sample['global_map']
    init_particles = data_sample['init_particles']

    plt.imshow(global_map[0, :, :, 0])
    plt.scatter(true_states[0, 0, 0], true_states[0, 0, 1], s=20, c='#7B241C', alpha=.75)
    plt.scatter(init_particles[0, :, 0], init_particles[0, :, 1], s=10, c='#515A5A', alpha=.25)

    plt.show()

    print('display done')

if __name__ == '__main__':
    params = arguments.parse_args()

    display_data(params)
