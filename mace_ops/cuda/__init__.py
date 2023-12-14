import os
import torch
import sysconfig

_HERE = os.path.realpath(os.path.dirname(__file__))
EXT_SUFFIX = sysconfig.get_config_var('EXT_SUFFIX')


torch.ops.load_library(_HERE + '/symmetric_contraction.so')
torch.ops.load_library(_HERE + '/invariant_message_passing_old.so')
torch.ops.load_library(_HERE + '/invariant_message_passing.so')
torch.ops.load_library(_HERE + '/equivariant_message_passing.so')

#torch.ops.load_library(_HERE + '/linear.so')
torch.ops.load_library(_HERE + '/linear_wmma.so')
#torch.ops.load_library(_HERE + '/embedding_tools.so')