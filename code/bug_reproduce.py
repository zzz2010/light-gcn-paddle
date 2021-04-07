

import paddorch
from paddorch.convert_pretrain_model import load_pytorch_pretrain_model
from glob import glob

from paddle import fluid
import os
import numpy as np
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





if __name__ == '__main__':
  from paddle import fluid

  input_weight_file = "../torch_pretrained_models/lgn-%s-3-64.pth.tar"%world.dataset

  # ==============================
  utils.set_seed(world.seed)
  print(">>SEED:", world.seed)
  # ==============================

  N_users=4000
  user_list=np.arange(N_users)
  item_list=np.arange(N_users)
  neg_list=np.arange(N_users,N_users*2)

  place = fluid.CUDAPlace(0)
  with fluid.dygraph.guard(place=place):

      Recmodel_paddle=model.LightGCN(world.config,dataset)
      Recmodel_paddle.load_state_dict(paddorch.load("../Recmodel_paddle_state_dict"))
      opt = paddorch.optim.Adam(Recmodel_paddle.parameters())

      print("loaded paddle")



      paddle_out=Recmodel_paddle(paddorch.LongTensor(user_list),paddorch.LongTensor(item_list))

      paddle_out,_ =Recmodel_paddle.bpr_loss(paddorch.LongTensor(user_list),paddorch.LongTensor(item_list),paddorch.LongTensor(neg_list))



      opt.zero_grad()
      paddle_loss=paddle_out.sum().backward( )

      paddle_grad=Recmodel_paddle.embedding_user.weight.gradient()
      print("paddle grad:",np.mean(paddle_grad),np.max(paddle_grad),np.min(paddle_grad))

      paddle_grad = Recmodel_paddle.embedding_item.weight.gradient()
      print("paddle grad:",np.mean(paddle_grad),np.max(paddle_grad),np.min(paddle_grad))





