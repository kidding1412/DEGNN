import json
import os

path = r'/home/dj/gnn2/dataset/new_train/tar-1.30_clang-5.0_x86_64_O0_rmt_c_isalpha_cpio-2.12_gcc-4.9.4_arm_32_O0_rmt_c_isalpha.json'
with open(path, 'r') as f:
    data = json.load(f)
print(data)

# path = path.split('/')[-1][:-5]
# print(path)
# binname1 = path.split('_')[0]
# print(binname1)
# list1 = path.split(binname1)
# print(list1)
# list1.remove('')
# print(list1)
# list2 = list1[0].split('_')
# list3 = list1[1].split('_')
# print(list2)
# print(list3)
# print(list2[5:])
# print(list3[5:])
# if list2[5:] == list3[5:]:
#     print(1)
# else:
#         print(0)