import world
import utils
from world import cprint
import paddorch as torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
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
    Recmodel = register.MODELS[world.model_name](world.config, dataset)


    Recmodel = Recmodel.to(world.device)
    bpr = utils.BPRLoss(Recmodel, world.config)

    weight_file = utils.getFileName()
    print(f"load and save to {weight_file}")
    if world.LOAD:
        try:
            Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
            world.cprint(f"loaded model weights from {weight_file}")
        except :
            print(f"{weight_file} not exists, start from beginning")
    Neg_k = 1

    # init tensorboard
    if world.tensorboard:
        w : SummaryWriter = SummaryWriter(
                                        join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                        )
    else:
        w = None
        world.cprint("not enable tensorflowboard")

    if world.config['single']:
        world.cprint("LightGCN-Single mode")

    try:
        last_best_recall=0
        current_recall_20=0
        for epoch in range(world.TRAIN_epochs):

            start = time.time()
            if epoch %10 == 0:
                cprint("[TEST]|"+world.dataset)
                results=Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
                current_recall_20=results['recall'][0]
            output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
            print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
            if last_best_recall < current_recall_20:
                last_best_recall=current_recall_20
                print("save model at EPOCH",epoch)
                torch.save(Recmodel.state_dict(), weight_file)
    finally:
        if world.tensorboard:
            w.close()