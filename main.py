from chainer import optimizer, optimizers
import chainer.functions as F
from agent import Model, objective_function_for_policy, objective_function_for_value
from reversi import Board
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import random
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

batchsize = 32
cursor = 0
Memory_s = np.ndarray(shape=(50000,2,8,8), dtype=np.float32)
Memory_pz = np.ndarray(shape=(50000,2), dtype=np.int32)
model = Model()
optimizer = optimizers.MomentumSGD()
optimizer.setup(model)
#optimizer.add_hook(optimizer.WeightDecay(0.0005))

for epoch in tqdm(range(5)):
    for game in range(10):
        env = Board()
        #env.render()
        player = 1
        first_cursor = cursor
        candidates = env.candidates(player)
        while len(candidates) > 0 or len(env.candidates(player*-1)) > 0:
            if len(candidates) == 0:
                player *= -1
                continue
            values = np.array([0 for i in range(len(candidates))], dtype=np.float32)
            for i in range(len(candidates)):
                env_ = deepcopy(env)
                env_.step(candidates[i],player)
                obs = np.zeros(shape=(1,2,8,8), dtype=np.float32)
                obs[0] = env_.convert(player)
                values[i] = model(obs).data[0]
            pos = candidates[np.argmax(values)]
            Memory_s[cursor%50000] = env.convert(player)
            env.step(pos, player)
            Memory_pz[cursor%50000][0] = pos
            Memory_pz[cursor%50000][1] = player
            cursor += 1
            player *= -1
            candidates = env.candidates(player)
            #env.render()
        winner = env.winner()
        if winner != 1:
            c = first_cursor
            while c%50000 != cursor:
                Memory_pz[c%50000][1] *= winner
                c += 1
    for step in range(10):
        x = np.zeros(shape=(32,2,8,8), dtype=np.float32)
        t = np.zeros(shape=(32,1), dtype=np.float32)
        index = list(np.random.randint(0,min(cursor,50000),batchsize))
        for i in range(batchsize):
            x[i] = Memory_s[index[i]]
            t[i] = Memory_pz[index[i]][1]
        model.cleargrads()
        y = model(x)
        loss = F.mean_squared_error(y,t)
        loss.backward()
        optimizer.update()
