

import paddorch
from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
from glob import glob

from paddle import fluid
import os
import numpy as np
import torch_model
import world
import utils
from world import cprint
import paddorch as torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import model
import Procedure
from os.path import join

import register
from register import dataset
import torch_dataloader





if __name__ == '__main__':
  import torch
  from paddle import fluid

  input_weight_file = "../torch_pretrained_models/lgn-%s-3-64.pth.tar"%world.dataset

  # ==============================
  utils.set_seed(world.seed)
  print(">>SEED:", world.seed)
  # ==============================
  place=fluid.CPUPlace()
  user_list=[1,2,3]
  item_list=[1,2,3]
  with fluid.dygraph.guard(place=place):
      torch_dataset=torch_dataloader.Loader(path="../data/"+world.dataset)
      Recmodel_torch=torch_model.LightGCN(world.config,torch_dataset)
      torch_state_dict=torch.load(input_weight_file, map_location=torch.device('cpu'))
      Recmodel_torch.load_state_dict(torch_state_dict)

      print("loaded torch")
      torch_users=torch.LongTensor(user_list )

      torch_out=Recmodel_torch(torch_users,torch.LongTensor(item_list))
      print(torch_out)

      Recmodel_paddle=model.LightGCN(world.config,dataset)
      paddle_state_dict=load_pytorch_pretrain_model(Recmodel_paddle,torch_state_dict)
      Recmodel_paddle.load_state_dict(paddle_state_dict)
      print("loaded paddle")



      paddle_out=Recmodel_paddle(paddorch.LongTensor([1,2,3]),paddorch.LongTensor([1,2,3]))
      print(paddle_out)

      assert max(torch_out.detach().numpy()-paddle_out.detach().numpy())<0.0001,"paddle and torch forward output not match!"

      torch_out.sum().backward()

      print("torch grad:",torch.mean(Recmodel_torch.embedding_user.weight.grad),torch.max(Recmodel_torch.embedding_user.weight.grad),torch.min(Recmodel_torch.embedding_user.weight.grad))

      paddle_loss=paddle_out.sum().backward(retain_graph=True)

      print("paddle grad:",np.mean(Recmodel_paddle.embedding_user.weight.gradient()),np.max(Recmodel_paddle.embedding_user.weight.gradient()),np.min(Recmodel_paddle.embedding_user.weight.gradient()))




