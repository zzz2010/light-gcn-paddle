import world
import utils
from world import cprint
import paddorch as torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join

###example:
###python eval_model.py --dataset yelp2018 --path checkpoints/lgn-yelp2018-3-64-2048-0.001.pth.tar.pdparams
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
from paddle import fluid

if world.config['device']=="cpu":
    place = fluid.CPUPlace()
else:
    place = fluid.CUDAPlace(0)
#
with fluid.dygraph.guard(place=place):
    model_fn=world.PATH

    Recmodel = register.MODELS[world.model_name](world.config, dataset)

    Recmodel.load_state_dict(torch.load(model_fn, map_location=torch.device('cpu')))


    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    world.cprint(f"loaded model weights from {model_fn}")

    Neg_k = 1

    results = Procedure.Test(dataset, Recmodel, 0, None, world.config['multicore'])
    ##result contains metric for top-K recommendation, K=20 by default
    cprint("[TEST]")
    print("DataSet:"+world.dataset)
    print("Recall@20: %.4f"%results['recall'][0])
    print("NDCG@20: %.4f" % results['ndcg'][0])
    print("Precision@20: %.4f" % results['precision'][0])

