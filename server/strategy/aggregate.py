# Copyright 2020 Adap GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Aggregation functions for strategy implementations."""


from functools import reduce
from typing import List, Tuple
import json

import numpy as np

from flwr.common import Weights


def aggregate(results: List[Tuple[Weights, int]], c1_fid, c2_fid, c3_fid, fid_bias) -> Weights:
    """Compute weighted average."""
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])
    ## num_examples_total = 각 클라이언트의 데이터셋 모두 더한것 : 여기서는 30000
    ## 21000이 된다면? 
    # Create a list of weights, each multiplied by the related number of examples
    ## 여기서 weight에 각 client별 데이터셋을 곱한 값으로 만든다 (여기서는 데이터셋의 크기를 가중치로 둔것임)
    ### results안에 weights와 dataset개수가있는데, 이 weight에 대해서 layer이라는 별명을 붙이고, 이 layer에 dataset의
    ### 개수를 곱한것이 weighted_weights list에 들어가게 된다. 
    clients_weight = [w for w,_ in results]
    # np.savez('/home/mjkim1/workspace/fl_gan/temp.npy', y)
    ##여기서 client별 weight는 clients_weight[n] 으로 나타낼수 있다
    ### Generator = y[n][:163] !! 검증끝남

    
    
    # print(f"weights 개수 : {len(y),}, dataset의 개수 : {x}")
    ## 1차로 number에 따른 bias 할당
    ### weighted_weights에는 3개의 공간이 존재하고, 각 공간에는 client별 weight*bias가 들어가있다.
    weighted_weights = [
        [layer * num_examples for layer in weights] for weights, num_examples in results
    ]
    # print(len(weighted_weights))
    '''fid bias implements
    만약 10, 15, 20 이면 1/10 곱하고 1/15 곱하고 1/20 곱하고 최종에서 x = 1 / (0.1 + 0.15 + 0.2))   
    '''
    
    final_c1 = c1_fid * fid_bias
    final_c2 = c2_fid * fid_bias
    final_c3 = c3_fid * fid_bias
    fid_bias_weights = []
    fid_bias_weights[0] = fid_bias_weights.append(weighted_weights[0])
    fid_bias_weights[1] = fid_bias_weights.append(weighted_weights[1])
    fid_bias_weights[2] = fid_bias_weights.append(weighted_weights[2])
    fid_bias_weights[0] = [i * final_c1 for i in weighted_weights[0]]
    fid_bias_weights[1] = [i * final_c2 for i in weighted_weights[1]]
    fid_bias_weights[2] = [i * final_c3 for i in weighted_weights[2]]
    # print(fid_bias_weights[0])
    # Compute average weights of each layer
    ## weights_prime에는 이걸 평균낸 avg값이 들어가있고 공간은 1로 바뀌며, G+D 의 weight가 들어가있다.

    weights_prime: Weights = [
        (reduce(np.add, layer_updates) / (num_examples_total)) * 3.0
        for layer_updates in zip(*fid_bias_weights)
    ]
    # weights_prime: Weights = [
    #     reduce(np.add, layer_updates) / num_examples_total
    #     for layer_updates in zip(*weighted_weights)
    # ]
    #print(len(weights_prime))
    
    
    return weights_prime

def cal_G(results: List[Tuple[Weights, int]]) -> Weights:

    clients_w = [w for w,_ in results]
    client1_G_w = clients_w[0][:163]
    client2_G_w = clients_w[1][:163]
    client3_G_w = clients_w[2][:163]

    return client1_G_w, client2_G_w, client3_G_w

def weighted_loss_avg(results: List[Tuple[int, float]]) -> float:
    """Aggregate evaluation results obtained from multiple clients."""
    num_total_evaluation_examples = sum([num_examples for num_examples, _ in results])
    weighted_losses = [num_examples * loss for num_examples, loss in results]
    return sum(weighted_losses) / num_total_evaluation_examples


def aggregate_qffl(
    weights: Weights, deltas: List[Weights], hs_fll: List[Weights]
) -> Weights:
    """Compute weighted average based on  Q-FFL paper."""
    demominator = np.sum(np.asarray(hs_fll))
    scaled_deltas = []
    for client_delta in deltas:
        scaled_deltas.append([layer * 1.0 / demominator for layer in client_delta])
    updates = []
    for i in range(len(deltas[0])):
        tmp = scaled_deltas[0][i]
        for j in range(1, len(deltas)):
            tmp += scaled_deltas[j][i]
        updates.append(tmp)
    new_weights = [(u - v) * 1.0 for u, v in zip(weights, updates)]
    return new_weights
