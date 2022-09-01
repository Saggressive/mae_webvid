import torch
import sys
sys.path.append("../")
import models_cmae as models_mae
if __name__=="__main__":
    # model_path="/nlp_group/wuxing/suzhenpeng/mae/webvid_output_dir/checkpoint-0.pth"
    # model=torch.load(model_path,map_location="cpu")["model"]
    # print(model.keys())
    # for name,parameters in model.named_parameters():
    #     print(name)

    model = models_mae.__dict__["mae_vit_base_patch16"](norm_pix_loss=False)
    print(model)