from typing import List, Optional, Tuple, Union
import torch
from torch import nn
import math
from LLH import Forward_LLH
from BaseModel import TorchBaseModel

# Embedding layer for complex-valued embeddings for the diagonalized SSM matrices
class ComplexEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ComplexEmbedding, self).__init__()
        self.real_embedding = nn.Embedding(*args, **kwargs)
        self.imag_embedding = nn.Embedding(*args, **kwargs)
        
        self.real_embedding.weight.data *= 1e-3
        self.imag_embedding.weight.data *= 1e-3
        
    def forward(self, x):
        return torch.complex(self.real_embedding(x), self.imag_embedding(x))
    
class ScaledSoftPlus(nn.Module):
    def __init__(self, num_marks, threshold=20.):
        super(ScaledSoftPlus, self).__init__()
        self.threshold = threshold
        self.log_beta = nn.Parameter(torch.zeros(num_marks), requires_grad=True)
        
    def forward(self, x):
        beta = self.log_beta.exp()
        beta_x = beta * x
        return torch.where( # if above threshold, then the transform is effectively linear
            beta_x <= self.threshold, 
            torch.log1p(beta_x.clamp(max=math.log(1e5)).exp()) / beta, 
            x
        )
        
class IntensityNet(nn.Module):
    def __init__(self, input_dim, num_event_types, bias_bool):
        super().__init__()
        self.intensity_net = nn.Linear(input_dim, num_event_types, bias=bias_bool)
        self.softplus = ScaledSoftPlus(num_event_types)
        
    def forward(self, x):
        return self.softplus(self.intensity_net(x))
    
class S2P2(TorchBaseModel):
    def __init__(self, model_config):
        super(S2P2, self).__init__()
        self.n_layers = model_config.num_layers
        self.P = model_config.model_specs["P"]
        self.H = model_config.model_specs["H"]
        self.beta = model_config.model_specs.get("beta", 1.0)
        self.bias = model_config.model_specs.get("bias", True)
        
        layer_kwargs = dict(
            P=self.P,
            H=self.H,
            dropout_rate=model_config.model_specs.get("dropout_rate", 0.0),
            pre_norm=model_config.model_specs.get("pre_norm", True),
            post_norm=model_config.model_specs.get("post_norm", False),
            relative_time=model_config.model_specs.get("relative_time", False)
        )
        
        self.layers = nn.ModuleList([Forward_LLH(**layer_kwargs, is_first_layer = i == 0) for i in range(self.layers)])
        
        # One embedding to share amongst layers to be used as input into a layer-specific and input-aware impulse
        self.layers_mark_emb = nn.Embedding(self.num_event_types_pad, self.H)
        
        self.intensity_net = self.IntensityNet(input_dim = self.H, 
                                               num_event_types = self.num_event_types, 
                                               bias_bool = self.bias)
        
    
    # assumes time has already been evolved, taking a vertical stack of hidden states and computing intensities    
    def _get_intensity(self, x_LP: Union[torch.tensor, List[torch.tensor]], right_us_BNH) -> torch.Tensor:
        left_u_H = None
        for i, layer in enumerate(self.layers):
            if isinstance (x_LP, list):
                left_u_H = layer.depth_pass(x_LP[i], current_left_u_H = left_u_H, prev_right_u_H = right_us_BNH[i])
            else:
                left_u_H = layer.depth_pass(x_LP[..., i, :], current_left_u_H = left_u_H, prev_right_u_H = right_us_BNH[i])
        return self.intensity_net(left_u_H) # calls IntensityNet's forward() by pytorch's nn.Module implementation of __call__
        
    def _evolve_and_get_intensity_at_sampled_dts(self, x_LP, dt_G, right_us_H):
        left_u_GH = None
        for i, layer in enumerate(self.layers):
            x_GP = layer.get_left_limit(
                right_limit_P = x_LP[..., i, :],
                dt_G = dt_G,
                next_left_u_GH = left_u_GH,
                current_right_u_H = right_us_H[i]
            )
            left_u_GH = layer.depth_pass(
                current_left_x_P = x_GP,
                current_left_u_H = left_u_GH,
                prev_right_u_H = right_us_H[i]
            ) 
        return self.intensity_net(left_u_GH)
    
    def forward(self, batch, x_0_BLP: Optional[torch.Tensor] = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        t_BN, dt_BN, marks_BN, batch_non_pad_mask, _ = batch # TODO: find out where batch structure lives
        
        right_xs_BNP = [] # list over layers: each is [B,N,P], including both t_0 and t_N
        left_xs_BNm1P = [] # list over layers: each is [B,N-1,P] (forward variant)
        right_us_BNH = [None] # list over layers+1: per layer right u, element 0 is None as this is the 'input' to the first layer
        left_u_BNH, right_u_BNH = None, None # current layer's input as layer depth is traversed, initialized to None for first layer
        α_BNH = self.layers_mark_emb(marks_BN) # this is the mark impulse embedding \alpha_k_i for each event in the batch
        
        for i, layer in enumerate(self.layers):
            x_0 = x_0_BLP[:, i] if x_0_BLP is not None else None
            x_BNP, next_layer_left_u_BNH, next_layer_right_u_BNH = layer.forward(left_u_BNH, right_u_BNH, α_BNH, dt_BN, x_0)
            assert next_layer_right_u_BNH is not None
            right_xs_BNP.append(x_BNP)
            
            # if NOT backward variant, compute left-limit x(t_i-) from right-limit x(t_{i-1}+)
            # here we are creating a list of left-limit xs at each event time except the first (t_1-, t_2-, ..., t_N-) using
            # the .get_left_limit() method of the layer, which evolves the right-limit state at t_{i-1}+ forward by dt_i 
            # to get left-limit state at t_i-.
            if next_layer_left_u_BNH is None: # NOT backward variant
                left_xs_BNm1P.append(
                    layer.get_left_limit(
                        x_BNP[..., :-1, :], # at time [t_0, t_1, ..., t_{N-1}]
                        dt_BN[..., 1:].unsqueeze(-1), # with dts [t1-t0, t2-t1, ..., t_N-t_{N-1}]
                        current_right_u_H = right_u_BNH if right_u_BNH is None else right_u_BNH[..., :-1, :], # at times [t_0, t_1, ..., t_{N-1}]
                        next_left_u_H = left_u_BNH if left_u_BNH is None else left_u_BNH[..., 1:, :].unsqueeze(-2) # at times [t_1, t_2 ..., t_N]
                    ).squeeze(-2) # get it back to shape [B,N-1,P] by removing the singleton dimension added in get_left_limit
                )
            right_us_BNH.append(next_layer_right_u_BNH)
            left_u_BNH, right_u_BNH = next_layer_left_u_BNH, next_layer_right_u_BNH
            
        right_xs_BNLP = torch.stack(right_xs_BNP, dim = -2)
        
        ret_val = {
            "right_xs_BNLP": right_xs_BNLP, # [t_0, ..., t_N]
            "right_us_BNH": right_us_BNH # [t_0, ..., t_N]; list starting with None
        }
        if left_u_BNH is not None: # backward variant
            ret_val["left_u_BNm1H"] = left_u_BNH[..., 1:, :]
        else: # NOT backward variant
            ret_val["left_xs_BNm1LP"] = torch.stack(left_xs_BNm1P, dim = -2)
            
        # 'seq_len - 1' left limit for [t_1, ..., t_N] for events (u if available, x if not)
        # 'seq_len' right limit for [t_0, t_1, ..., t_{N-1}, t_N] for events xs or us
        return ret_val
    
    def loglike_loss(self, batch, **kwargs):
        # This method is a sort of "orchestration" for the log likelihood specific to S2P2, which takes into account right/left limits,
        # whereas torch_basemodel.compute_loglikelihood is a more general MATHEMATICAL method that just computes the log-likelihood.
        # loglike_loss does the S2P2 forward pass, computes intensities at event times and sampled times, and the passes these agruments
        # into torch_basemodel.compute_loglikelihood to get the actual log-likelihood values for the S2P2 model.
        # hidden states at the left and right limits around event time; note for the shift by 1 in indices:
        # consider a sequence [t0, t1, ..., tN]
        # Produces the following:
        # left_x: x0, x1, x2, ... <-> x_{t_1-}, x_{t_2-}, x_{t_3-}, ..., x_{t_N-} (note the shift in indices) for all layers
        #    OR ==>               <-> u_{t_1-}, u_{t_2-}, u_{t_3-}, ..., u_{t_N-} for last layer
        #
        # right_x: x0, x1, x2, ... <-> x_{t_0+}, x_{t_1+}, ..., x_{t_N+} for all layers
        # right_u: u0, u1, u2, ... <-> u_{t_0+}, u_{t_1+}, ..., u_{t_N+} for all layers
        forward_results = self.forward(batch)  # N minus 1 comparing with sequence lengths
        right_xs_BNLP, right_us_BNH = (
            forward_results["right_xs_BNLP"],
            forward_results["right_us_BNH"],
        )
        right_us_BNm1H = [
            None if right_u_BNH is None else right_u_BNH[:, :-1, :]
            for right_u_BNH in right_us_BNH
        ]

        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # evaluate intensity values at each event *from the left limit*, _get_intensity: [LP] -> [M]
        # left_xs_B_Nm1_LP = left_xs_BNm1LP[:, :-1, ...]  # discard the left limit of t_N
        # Note: no need to discard the left limit of t_N because "marks_mask" will deal with it
        if "left_u_BNm1H" in forward_results:  # ONLY backward variant
            intensity_B_Nm1_M = self.intensity_net(
                forward_results["left_u_BNm1H"]
            )  # self.ScaledSoftplus(self.linear(forward_results["left_u_BNm1H"]))
        else:  # NOT backward variant
            intensity_B_Nm1_M = self._get_intensity(
                forward_results["left_xs_BNm1LP"], right_us_BNm1H
            )

        # sample dt in each interval for MC: [batch_size, num_times=N-1, num_mc_sample]
        # N-1 because we only consider the intervals between N events
        # G for grid points
        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])

        # evaluate intensity at dt_samples for MC *from the left limit* after decay -> shape (B, N-1, MC, M)
        intensity_dts_B_Nm1_G_M = self._evolve_and_get_intensity_at_sampled_dts(
            right_xs_BNLP[
                :, :-1
            ],  # x_{t_i+} will evolve up to x_{t_{i+1}-} and many times between for i=0,...,N-1
            dts_sample_B_Nm1_G,
            right_us_BNm1H,
        )

        event_ll, non_event_ll, num_events, mark_ll, time_ll_pos = (
            self.compute_loglikelihood(
                lambda_at_event=intensity_B_Nm1_M,
                lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
                time_delta_seq=dts_BN[:, 1:],
                seq_mask=batch_non_pad_mask[:, 1:],
                type_seq=marks_BN[:, 1:],
            )
        )

        # compute extra statistics
        time_ll = time_ll_pos - non_event_ll

        # compute loss to optimize
        loss = -(event_ll - non_event_ll).sum()

        return_raw_ll = kwargs.get("return_raw_ll", False)
        res_dict = (
            {"non_event_ll": non_event_ll, "mark_intensity": intensity_B_Nm1_M}
            if return_raw_ll
            else None
        )

        return loss, num_events, mark_ll.sum(), time_ll.sum(), res_dict

    def compute_intensities_at_sample_times(
        self, event_times_BN, inter_event_times_BN, marks_BN, sample_dtimes, **kwargs
    ):
        """Compute the intensity at sampled times, not only event times.  *from the left limit*
        
        This method is a public-ish method used for prediction/simulation taht is passed into the thinning-based sampler:
        EventSampler.draw_next_time_one_step(). It's a higher-level wrapper whose job is essentially:
        "given the observed history (times & marks) and some propsed sample offsets, return intensities at those sample offsets"

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get("compute_last_step_only", False)

        # assume inter_event_times_BN always starts from 0
        _input = event_times_BN, inter_event_times_BN, marks_BN, None, None

        # 'seq_len - 1' left limit for [t_1, ..., t_N]
        # 'seq_len' right limit for [t_0, t_1, ..., t_{N-1}, t_N]

        forward_results = self.forward(
            _input
        )  # N minus 1 comparing with sequence lengths
        right_xs_BNLP, right_us_BNH = (
            forward_results["right_xs_BNLP"],
            forward_results["right_us_BNH"],
        )

        if (
            compute_last_step_only
        ):  # fix indices for right_us_BNH: list [None, tensor([BNH]), ...]
            right_us_B1H = [
                None if right_u_BNH is None else right_u_BNH[:, -1:, :]
                for right_u_BNH in right_us_BNH
            ]
            sampled_intensity = self._evolve_and_get_intensity_at_sampled_dts(
                right_xs_BNLP[:, -1:, :, :], sample_dtimes[:, -1:, :], right_us_B1H
            )  # equiv. to right_xs_BNLP[:, -1, :, :][:, None, ...]
        else:
            sampled_intensity = self._evolve_and_get_intensity_at_sampled_dts(
                right_xs_BNLP, sample_dtimes, right_us_BNH
            )
        return sampled_intensity  # [B, N, MC, M]