import sys
import os
import glob

conf_classes = {'T0':19,'T1':15,'T2':10,'T3':15,'T4':10,'T5':10,'T6':10}
conf_classes_v2 = {'19-1':19, '15-5':15, '10-5':10, '15-1':15, '10-10':10, '10-2':10, '10-1':10}
conf_dict = {'T0':'19-1','T1':'15-5','T2':'10-5','T3':'15-1','T4':'10-10','T5':'10-2','T6':'10-1'}
joint_with_bg = 71.0

def process_one_file(file_path, file_output, write=True):

    row_list = []
    with open(file_path,'r') as f:
        for row in f:
            row_list.append(row)

    row_list.reverse()
    for i,row in enumerate(row_list):
        if 'class_iou' in row:
            data_string = row_list[i-1] + row_list[i-2]
            break

    try:
        data_string = data_string.strip()[2:-2]
    except UnboundLocalError:
        print('File {} not of the correct format. Skipped'.format(file_path))
        return
    values = list(map(float, data_string.split('\', \'')))

    for conf in conf_dict.keys():
        if conf+'_' in file_path:
            n_classes = conf_dict[conf]
            break
        # raise ValueError
    # print('Arguments: \nstep 0 num classes (no background):{} \nlist of values: {}'.format(n_classes,values))

    n_old_classes = None
    for conf in conf_classes:
        if conf + '_' in file_path: n_old_classes = conf_classes[conf]
    for conf in conf_classes_v2:
        if conf in file_path: n_old_classes = conf_classes_v2[conf]
    if n_old_classes is None:
        n_old_classes = int(sys.argv[1])
        print('Old classes manually set to: {} \n'.format(n_old_classes))

    mIoU_old = sum(values[1:n_old_classes+1]) / len(values[1:n_old_classes+1]) * 100
    mIoU_new = sum(values[n_old_classes+1:]) / len(values[n_old_classes+1:]) * 100
    mIoU_with_bg = sum(values) / len(values) * 100
    mIoU_without_bg = sum(values[1:]) / len(values[1:]) * 100
    delta_with_bg = mIoU_with_bg - joint_with_bg


    s = ''
    for conf in conf_dict.keys():
        if conf + '_' in file_path: s += (conf_dict[conf] + '   ')
    if 'gan' in file_path: s += 'GAN\n'
    else: s += 'FLICKR\n'

    s_with_bg = ''
    s += 'perClassIoU (no bg):\n'
    s_with_bg += 'perClassIoU (with bg):\n'
    if 'gan' in file_path:
        s += 'GAN & '
        s_with_bg += 'GAN & '
    else:
        s += 'Web & '
        s_with_bg += 'Web & '
    s += '  & {:.1f} & {:.1f} & {:.1f}'.format(mIoU_old, mIoU_new, mIoU_without_bg)
    s_with_bg += '  & {:.1f} & {:.1f} & {:.1f} & {:.1f}'.format(mIoU_old, mIoU_new, mIoU_with_bg,delta_with_bg )


    if write:
        with open(file_output, 'a') as f:
            f.write(file_path + '\n' + s + '\n')
            f.write(s_with_bg + '\n\n')


dir = 'processed/'
dir_path = 'outputs/'
file_out = 'outputs/' + dir + 'output.txt'
if os.path.exists(file_out): os.remove(file_out)
for file in glob.glob(f'{dir_path}*.txt'):
    process_one_file(file, file_out)

print('Output file: {}'.format(file_out))
