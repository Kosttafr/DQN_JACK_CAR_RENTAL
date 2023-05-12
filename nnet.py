import torch.nn as nn
import torch.nn.functional as F

INNER_LAYER_N = 512


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, INNER_LAYER_N)
        self.layer2 = nn.Linear(INNER_LAYER_N, INNER_LAYER_N)
        self.layer3 = nn.Linear(INNER_LAYER_N, INNER_LAYER_N)
        self.layer4 = nn.Linear(INNER_LAYER_N, INNER_LAYER_N)
        self.layer5 = nn.Linear(INNER_LAYER_N, n_actions)
#--------------------------------------------------
        # self.layer1 = nn.Linear(n_observations, INNER_LAYER_N)
        # self.layer2 = nn.Linear(INNER_LAYER_N, INNER_LAYER_N)
        # self.layer3 = nn.Linear(INNER_LAYER_N, INNER_LAYER_N)
        # self.layer4 = nn.Linear(INNER_LAYER_N, INNER_LAYER_N)
        # self.layer5 = nn.Linear(INNER_LAYER_N, n_actions)
#--------------------------------------------------
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
#--------------------------------------------------
        # x = F.tanh(self.layer1(x))
        # x = F.tanh(self.layer2(x))
        # x = F.tanh(self.layer3(x))
        # x = F.tanh(self.layer4(x))
#--------------------------------------------------
        return self.layer5(x)