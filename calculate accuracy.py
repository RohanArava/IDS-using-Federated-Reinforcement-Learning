import numpy as np
import json
folder = input("Enter folder path:")
accs = []
losses = []
for i in range(0, 10):
    agent = str(np.load(f"{folder}/client-{i}-metrics.npy", allow_pickle=True))
    acclss = json.loads(agent.replace("'",'"'))
    accs.append(acclss["accuracy"][-1])
    losses.append(acclss["loss"][-1])
print(np.mean(accs))
print(np.mean(losses))