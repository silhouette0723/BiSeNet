import os
import torch
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import bv2_214MB as bv2

def profile_onnx_model(model_path):
    import onnx_tool
    m = onnx_tool.Model(model_path)
    m.graph.shape_infer({'src': np.zeros((1,3,160,96)),
                         'r2i':np.zeros((1,12,160,96)),
                         'r3i':np.zeros((1,32,160,96)),
                         'r4i':np.zeros((1,64,160,96))})
    m.graph.profile()
    m.graph.print_node_map('res/result.txt')
    m.graph.print_node_map()
    
def convert_bisenetv2():
    device = torch.device("cuda")
    # pth_model_path = "res/model_260MB_epoch_100.pth"
    model = bv2.BiSeNetV2(2).to(device)
    # model.load_state_dict(torch.load(pth_model_path, map_location=device, weights_only=True))
    model.eval()
    
    src = torch.rand(1, 3, 160, 96).cuda()
    input_names = ['input']
    output_names = ['output',]
    
    torch.onnx.export(model,(src,),'res/result.onnx',verbose=False,
                        input_names=input_names,
                        output_names=output_names,
                        opset_version=11)
    print("\n model convert done.\n")
    
convert_bisenetv2()
profile_onnx_model("res/result.onnx")

