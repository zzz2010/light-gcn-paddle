

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

  N_users=4000
  user_list=np.arange(N_users)
  item_list=np.arange(N_users)
  neg_list=np.arange(N_users,N_users*2)

  # import joblib
  #
  # user_list,item_list,neg_list=joblib.load("debug.dat.joblib")
  place = fluid.CUDAPlace(0)
  with fluid.dygraph.guard(place=place):
      torch_dataset=torch_dataloader.Loader(path="../data/"+world.dataset)
      Recmodel_torch=torch_model.LightGCN(world.config,torch_dataset)
      torch_state_dict=torch.load(input_weight_file, map_location=torch.device('cpu'))
      Recmodel_torch.load_state_dict(torch_state_dict)

      print("loaded torch")
      torch_users=torch.LongTensor(user_list )

      torch_out=Recmodel_torch(torch_users,torch.LongTensor(item_list))
      torch_out,_=Recmodel_torch.bpr_loss(torch_users,torch.LongTensor(item_list),torch.LongTensor(neg_list))


      Recmodel_paddle=model.LightGCN(world.config,dataset)
      opt = paddorch.optim.Adam(Recmodel_paddle.parameters())
      paddle_state_dict=load_pytorch_pretrain_model(Recmodel_paddle,torch_state_dict)
      Recmodel_paddle.load_state_dict(paddle_state_dict)
      paddorch.save(Recmodel_paddle.state_dict(),"Recmodel_paddle_state_dict")
      print("loaded paddle")



      paddle_out=Recmodel_paddle(paddorch.LongTensor(user_list),paddorch.LongTensor(item_list))
      paddle_out,_ =Recmodel_paddle.bpr_loss(paddorch.LongTensor(user_list),paddorch.LongTensor(item_list),paddorch.LongTensor(neg_list))
      print("forward output,max diff:",np.max(np.abs(torch_out.detach().numpy()-paddle_out.detach().numpy())))


      assert max(torch_out.detach().numpy()-paddle_out.detach().numpy())<0.0001,"paddle and torch forward output not match!"

      torch_out.sum().backward( )
      torch_grad=Recmodel_torch.embedding_user.weight.grad.detach().cpu().numpy()
      print("torch grad:",np.mean(torch_grad),np.max(torch_grad),np.min(torch_grad))

      opt.zero_grad()
      paddle_loss=paddle_out.sum().backward( )

      paddle_grad=Recmodel_paddle.embedding_user.weight.gradient()
      print("paddle grad:",np.mean(paddle_grad),np.max(paddle_grad),np.min(paddle_grad))
      print("maximum User grad difference",np.max(np.abs(torch_grad-paddle_grad)),paddle_grad.shape)

      torch_grad = Recmodel_torch.embedding_item.weight.grad.detach().cpu().numpy()
      print("torch grad:", np.mean(torch_grad), np.max(torch_grad), np.min(torch_grad))
      paddle_grad = Recmodel_paddle.embedding_item.weight.gradient()
      print("paddle grad:", np.mean(paddle_grad), np.max(paddle_grad), np.min(paddle_grad))
      print("maximum Item grad difference", np.max(np.abs(torch_grad - paddle_grad)), paddle_grad.shape)
      from  matplotlib import  pyplot as plt
      # plt.hist(np.abs(torch_grad - paddle_grad),bins=100)
      plt.scatter(torch_grad,np.abs(torch_grad - paddle_grad),s=1)
      plt.savefig("debug.png")
      plt.show()






