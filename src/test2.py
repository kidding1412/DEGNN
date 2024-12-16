import os

path1 = r'/home/dj/dataset/all_elf_file/'

file_list = [filename[:-4] for filename in os.listdir(path1) if filename.endswith('.elf')]

compiler_count = {}
architecture_count = {}
bit_count = {}
optimization_count = {}

for name in file_list:
    parts = name.split('_')
    compiler = parts[1]
    architecture = parts[2]
    bit = parts[3]
    optimization = parts[4]

    compiler_count[compiler] = compiler_count.get(compiler, 0) + 1
    architecture_count[architecture] = architecture_count.get(architecture, 0) + 1
    bit_count[bit] = bit_count.get(bit, 0) + 1
    optimization_count[optimization] = optimization_count.get(optimization, 0) + 1

print(f"编译器种类：{len(compiler_count)}")
print(f"编译器详细列表：{compiler_count}")
print(f"架构种类：{len(architecture_count)}")
print(f"架构详细列表：{architecture_count}")
print(f"位数种类：{len(bit_count)}")
print(f"位数详细列表：{bit_count}")
print(f"优化种类：{len(optimization_count)}")
print(f"优化详细列表：{optimization_count}")

