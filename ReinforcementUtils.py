import torch
import torch.nn as nn
import numpy as np
import random
import gc
from operator import itemgetter
from collections import deque
from MyUtils import get_tensor_loader
from Args import args

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, item):
        self.memory.append(item)
    def sample(self):
        mem = list(zip(*self.memory))
        weights = normalize(mem[2])
        length = len(self.memory)
        indices = np.arange(length)
        sample_indices = np.random.choice(indices, size=length//10, replace=False, p=weights)
        features = list(itemgetter(*sample_indices)(mem[0]))
        labels = list(itemgetter(*sample_indices)(mem[1]))
        return get_tensor_loader(features, labels)

    def __len__(self):
        return len(self.memory)

def reinforcement_train(net1, net2, train_loader):
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(net1.parameters(), lr=args.lr)
    memory = ReplayMemory(1000)
    net2.load_state_dict(net1.state_dict())
    for epoch in range(args.epochs):
        last = None
        for features, labels in train_loader:
            if last is None:
                last = (features, labels)
                continue
            next_state = features
            next_label = labels
            features = last[0]
            labels = last[1]
            outputs = net1(features)
            # q_values, _ = torch.max(outputs, dim=1)
            # predictions = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
            rewards = get_rewards(outputs, labels)
            targets = torch.add(rewards, torch.mul(get_rewards(net2(next_state), next_label), args.param_gamma))
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            last = (features, labels)
            target_sd = net2.state_dict()
            current_sd =  net1.state_dict()
            for key in current_sd:
                target_sd[key] = current_sd[key]*args.param_tau + target_sd[key]*(1-args.param_tau)
            net2.load_state_dict(target_sd)
            for feature, label, weight in zip(features, labels, get_probability_weights(outputs, labels)):
                memory.push((feature, label, weight))  
    sample = memory.sample()
    for epoch in range(args.epochs):
        for features, labels in sample:
            optimizer.zero_grad()
            outputs = net1(features)
            rewards = get_rewards(outputs, labels)
            loss = criterion(outputs, rewards.float())
            loss.backward()
            optimizer.step()
            


def get_probability_weights(outputs, labels):
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    
    weights = []
    for output_vector, label in zip(outputs, labels):
        target_vector = get_target_vector(label)
        error_vector = [abs(i-j) for i, j in zip(output_vector, target_vector)]
        error = sum(error_vector)
        weights.append(pow(error, args.per_exponent))
        
    return weights

def get_target_vector(label):
    if(label == 0):
        return [1, 0]
    
    if(label == 1):
        return [0, 1]

# def array_softmax(arr):
#     return np.exp(arr) / np.sum(np.exp(arr), axis=0)

# def get_targets(outputs, labels):
#     targets = []
#     for output, label in zip(outputs, labels):
        
#         output = output.detach().cpu().numpy()
#         label = label.detach().cpu().numpy()
#         # print(output)
#         prediction = np.argmax(array_softmax(output))
#         reward = get_reward(prediction, label)
        
#         if(reward == 1):
#             target = prediction
#         else:
#             target = label
        
#         targets.append(target)
        
#     return torch.from_numpy(np.asarray(targets)).float().type(torch.LongTensor)


def get_reward(prediction, label):
    if(prediction == label):
        return 1
    else:
        return 0

def get_rewards(outputs, labels):
    rewards = []
    for pred, lab in zip(outputs, labels):
        # p0 = 1 if pred[0] > pred[1] else 0
        # p1 = 0 if pred[0] > pred[1] else 1
        l = [1, 0] if lab == 0 else [0, 1]
        rewards.append(l)
    return torch.Tensor(rewards).float().type(torch.LongTensor)

def normalize(weights):
    total_sum = sum(weights)
    result = [x/total_sum for x in weights]
    return result        