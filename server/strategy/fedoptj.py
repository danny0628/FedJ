from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from training.networks import *
import dnnlib as dnnlib
import torch
import numpy as np
from collections import OrderedDict
import click
import PIL.Image
import os
import torchvision

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from .aggregate import aggregate, weighted_loss_avg, cal_G
from .fid_score import fid_value
from .strategy import Strategy
from .fedopt import FedOpt

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class FedOptJ(FedOpt):

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-9,
    ) -> None:
        """Federated Averaging strategy.

        Implementation based on https://arxiv.org/abs/1602.05629

        Parameters
        ----------
        fraction_fit : float, optional
            Fraction of clients used during training. Defaults to 0.1.
        fraction_eval : float, optional
            Fraction of clients used during validation. Defaults to 0.1.
        min_fit_clients : int, optional
            Minimum number of clients used during training. Defaults to 2.
        min_eval_clients : int, optional
            Minimum number of clients used during validation. Defaults to 2.
        min_available_clients : int, optional
            Minimum number of total clients in the system. Defaults to 2.
        eval_fn : Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
            Optional function used for validation. Defaults to None.
        on_fit_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure training. Defaults to None.
        on_evaluate_config_fn : Callable[[int], Dict[str, Scalar]], optional
            Function used to configure validation. Defaults to None.
        accept_failures : bool, optional
            Whether or not accept rounds containing failures. Defaults to True.
        initial_parameters : Parameters, optional
            Initial global model parameters.
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn]
            Metrics aggregation function, optional.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )

        if (
            min_fit_clients > min_available_clients
            or min_eval_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)



    def __repr__(self) -> str:
        rep = f"FedAvg(accept_failures={self.accept_failures})"
        return rep

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return the sample size and the required number of available
        clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients

    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_eval)
        return max(num_clients, self.min_eval_clients), self.min_available_clients

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        initial_parameters = self.initial_parameters
        self.initial_parameters = None  # Don't keep initial parameters in memory
        return initial_parameters

    def evaluate(
        self, parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights)
        if eval_res is None:
            return None
        loss, metrics = eval_res
        return loss, metrics

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_eval == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        ## results = 동형 암호화된 weight이므로 알아볼수 없음
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        # print(f"weights_results : {weights_results}")
        ## 각 clients의 G만 파라미터화 시킴
        client1_fid, client2_fid, client3_fid = cal_G(weights_results)
        netG1 = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=1)
        netG2 = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=1)
        netG3 = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=1)
        params_dict_1 = zip(netG1.state_dict().keys(), client1_fid)
        params_dict_2 = zip(netG2.state_dict().keys(), client2_fid)
        params_dict_3 = zip(netG3.state_dict().keys(), client3_fid)
        state_dict_1 = OrderedDict({k: torch.tensor(v) for k, v in params_dict_1})
        state_dict_2 = OrderedDict({k: torch.tensor(v) for k, v in params_dict_2})
        state_dict_3 = OrderedDict({k: torch.tensor(v) for k, v in params_dict_3})
        
        netG1.load_state_dict(state_dict_1)
        netG2.load_state_dict(state_dict_2)
        netG3.load_state_dict(state_dict_3)
        
        c1_path = '/home/mjkim1/workspace/fl_gan/stylegan2-ada-pytorch/clients_fake_img/c1'
        c2_path = '/home/mjkim1/workspace/fl_gan/stylegan2-ada-pytorch/clients_fake_img/c2'
        c3_path = '/home/mjkim1/workspace/fl_gan/stylegan2-ada-pytorch/clients_fake_img/c3'
        
        print("Generate Fake image Start")
        generate_images(network_pkl= netG1, seeds=num_range('0-1000'), truncation_psi=0.5, noise_mode='const', outdir=c1_path)
        generate_images(network_pkl= netG2, seeds=num_range('0-1000'), truncation_psi=0.5, noise_mode='const',outdir=c2_path)
        generate_images(network_pkl= netG3, seeds=num_range('0-1000'), truncation_psi=0.5, noise_mode='const',outdir=c3_path)
        
        ## fake 이미지 생성완료
        ## 이제 이 fake 이미지를 가지고 fid를 구해야됨. 1vs2 / 2vs3 / 1vs3 이렇게 총 3가지의 fid score 필요 
        # 만약, 19 15 30 이렇게 나왔으면 << 이부분은 고민좀 해보고 일단 fid 도출까지 해보자
        
        c1_fid = fid_value(path=[c1_path,c2_path])
        c2_fid = fid_value(path=[c2_path,c3_path])
        c3_fid = fid_value(path=[c3_path,c1_path])
         
        j_metric1 = 1 / (c1_fid + c3_fid)
        j_metric2 = 1 / (c2_fid + c1_fid)
        j_metric3 = 1 / (c3_fid + c2_fid)
        
        fid_bais = 1 / (j_metric1 + j_metric2 + j_metric3)

        print(f'c1&c2_fid :{c1_fid}, c2&c3_fid :{c2_fid}, c3&c1_fid :{c3_fid}, fid_bais :{fid_bais}')
        #final_c1 + final_c2 + final_c3 = 1.0
        # final_c1 = j_metric1*fid_bais
        # final_c2 = j_metric2*fid_bais
        # final_c3 = j_metric3*fid_bais
        
        ##
        print("Calculate FedJ Algorithm")
        parameters_aggregated = weights_to_parameters(aggregate(weights_results, j_metric1, j_metric2, j_metric3, fid_bais))
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
            
            

        fedj_weights_aggregate  = parameters_to_weights(parameters_aggregated)
        # Adam
        delta_t = [
            x - y for x, y in zip(fedj_weights_aggregate, self.current_weights)
        ]

        # m_t
        if not self.m_t:
            self.m_t = [np.zeros_like(x) for x in delta_t]
        self.m_t = [
            self.beta_1 * x + (1 - self.beta_1) * y for x, y in zip(self.m_t, delta_t)
        ]

        # v_t
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in delta_t]
        self.v_t = [
            self.beta_2 * x + (1 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, delta_t)
        ]

        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.m_t, self.v_t)
        ]

        self.current_weights = new_weights

        return weights_to_parameters(self.current_weights), metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_loss_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.evaluate_metrics_aggregation_fn:
            eval_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.evaluate_metrics_aggregation_fn(eval_metrics)
        elif rnd == 1:
            log(WARNING, "No evaluate_metrics_aggregation_fn provided")

        return loss_aggregated, metrics_aggregated


def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------
import re
def generate_images(
    network_pkl: str,
    seeds: num_range('0-1000'),
    truncation_psi: 0.5,
    noise_mode: 'const',
    outdir: str,
):


    device = torch.device('cuda')
    G = network_pkl.to(device)
    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)

    # Generate images.
    ws = {}
    for seed_idx, seed in enumerate(seeds):
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
        
        
        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        
        from torchvision.utils import save_image
        i=img.view(512,512)
        
        
        PIL.Image.fromarray(i.cpu().numpy()).save(f'{outdir}/seed{seed:04d}.png')