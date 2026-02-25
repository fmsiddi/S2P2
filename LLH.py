from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from HiPPO import DPLR_HiPPO

class Forward_LLH(nn.Module):
    
    def __init__(
        self,
        P: int,
        H: int,
        dropout_rate: float = 0.0,
        pre_norm: bool = True,
        post_norm: bool = False,
        is_first_layer: bool = False,
        relative_time: bool = False
    ):
        super(Forward_LLH, self).__init__()
        
        # inscribe config file parameters as class attributes
        self.P = P
        self.H = H
        self.dropout_rate = dropout_rate
        self.pre_norm = pre_norm
        self.post_norm = post_norm
        self.is_first_layer = is_first_layer
        self.relative_time = relative_time
        
        self.act_func = nn.Sequential(nn.GELU(), nn.Dropout(p=self.dropout_rate)) # D(GELU())
        self.norm = nn.LayerNorm(self.H) # we will normalize over the H features of the input and output
        
        # the multiplication by .001 is in line with the practice of avoiding uncontrolled intial state magnitude that can 
        # destabilize training. 
        self.x_0_P = nn.Parameter(torch.complex(torch.randn(self.P), torch.randn(self.P)) * 1e-3, 
                                  requires_grad=True)
        
        self._init_ssm_params()
        
    def _init_ssm_params(self):
        self._init_A()
        if not self.is_first_layer:
            self._init_B()
        self._init_C()
        if not self.is_first_layer:
            self._init_D()
        self._init.E()
        
    def _init_A(self):        
        Λ_P, V_PP, _, _ = DPLR_HiPPO(self.P)
        
        self.V_PP = V_PP
        self.V_star_PP = V_PP.conj().T
        
        # the real part of Λ is initialized to -0.5 across all P channels so it is guaranteed that log is receiving 
        # a positive input here
        self.log_neg_real_Λ_P = nn.Parameter((-Λ_P.real).log()) 
        self.imag_Λ_P = nn.Parameter(Λ_P.imag)
        
        if self.relative_time:
            self.Δ_net = nn.Linear(self.H, self.P, bias=True)
            with torch.no_grad():
                self.Δ_net.weight.copy_(nn.init.xavier_normal_(self.Δ_net.weight))
                
                bias = torch.ones(self.P)
                bias += torch.log(-torch.expm1(-bias))
                self.Δ_net.bias.copy_(bias)
        else:
            self.log_step_size_P = nn.Parameter(torch.zeros(size=(self.P)), requires_grad=False) # this may be unneeded
            
    @property
    def Λ_P(self):
        return torch.complex(-torch.exp(self.log_neg_real_Λ_P), self.imag_Λ_P)
    
    def _init_B(self):
        B = nn.init.xavier_normal_(torch.zeros((self.P, self.H)))
        B_tilde_PH = self.V_star_PP @ B.type(torch.complex64) # TODO: paper says we multiply by negative V*?
        self.B_tilde_PH = nn.Parameter(B_tilde_PH)
        
    def _init_C(self):       
        C = nn.init.xavier_normal_(torch.zeros((self.H, self.P)))
        C_tilde_HP = C.type(torch.complex64) @ self.V_PP
        self.C_tilde_HP = nn.Parameter(C_tilde_HP)
        
    def _init_D(self):        
        D_H = torch.zeros(self.H)
        nn.init.normal_(D_H, std=1.0)
        self.D_H = nn.Parameter(D_H)
        
    def _init_E(self):        
        E = nn.init.xavier_normal_(torch.zeros((self.P, self.H))) # R = H
        E_tilde_PH = self.V_star_PP @ E.type(torch.complex64)
        self.E_tilde_PH = nn.Parameter(E_tilde_PH)
        
    def compute_impulse(self,α_H):
        # although the paper defines α as a RxK (we set R = H) matrix, we are only taking a column vector from this matrix
        tilde_Eα_P = torch.einsum("ph,...h->...p", self.E_tilde_PH, α_H.type(torch.complex64))
        return tilde_Eα_P
    
    def get_Λ_i(self, right_u_NH, shift_u=True):
        if self.relative_time and (right_u_NH is not None): # right_u is null for first layer
            if shift_u:
                right_u_NH = F.pad(right_u_NH[..., :-1,:], (0,0,1,0)) # TODO: maybe explain this
            Λ_i_NP = F.softplus(self.Δ_net(right_u_NH)) * self.Λ_P
            return {"Λ_i_NP": Λ_i_NP} # these keys will be used in _ssm in order to determine the dimension
        else:
            if self.relative_time: # relative_time is true but it's the first layer (right_u is null)
                Λ_i_P = F.softplus(self.Δ_net.bias) * self.Λ_P # there is no u to multiply the weight matrix by
            else:
                Λ_i_P = self.Λ_P
            return {"Λ_i_P": Λ_i_P} # these keys will be used in _ssm in order to determine the dimension
        
    def forward(
        self,
        left_u_NH: Optional[torch.Tensor],
        right_u_NH: Optional[torch.Tensor],
        α_NH: torch.Tensor,
        dt_N: torch.Tensor, # [0, t1-t0, ..., t_N-t_{N-1}]
        x_0_P: Optional[torch.Tensor] = None # this argument may be omitted
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        *leading_dims, _, _ = α_NH.shape # leading_dims will typically just be batch size B
        num_leading_dims = len(leading_dims)
        
        # the state is P-dimensional, but it needs to be broacast across the B sequences 
        # basically we just need to reformat initial_state_P as a [...,P] tensor (most likely a [B,P] tensor)
        if x_0_P is None:
            x_0_P = self.x_0_P.view(*[1 for _ in range(num_leading_dims)], -1).expand(*leading_dims, -1)
        normed_left_u_NH = left_u_NH
        normed_right_u_NH = right_u_NH
        if normed_left_u_NH is not None:
            # checking that all the dimensions match
            assert all(u_d == a_d for u_d, a_d in zip(normed_left_u_NH.shape, α_NH.shape))
            if self.pre_norm:
                normed_left_u_NH = self.norm(normed_left_u_NH) # normed_left_u_NH is not actually norm'd until now
        if normed_right_u_NH is not None:
            # checking that all the dimensions match
            assert all(u_d == a_d for u_d, a_d in zip(normed_right_u_NH.shape, α_NH.shape))
            if self.pre_norm:
                normed_right_u_NH = self.norm(normed_right_u_NH) # normed_right_u_NH is not actually norm'd until now
        
        # this _ssm() call is what actually carries out the bulk of of Algorithm 1
        right_x_NP, left_y_NH, right_y_NH = self._ssm(
            left_u_NH = normed_left_u_NH,
            right_u_NH = normed_right_u_NH,
            tilde_Eα_NP = self.compute_impulse(α_NH),
            dt_N = dt_N,
            x_0_P = x_0_P
        )
        
        next_layer_left_u_NH = next_layer_right_u_NH = None
        if left_y_NH is not None:
            next_layer_left_u_NH = self.act_func(left_y_NH) + (left_u_NH if left_u_NH is not None else 0.0)
            if self.post_norm:
                next_layer_left_u_NH = self.norm(next_layer_left_u_NH)
        if right_y_NH is not None:
            next_layer_right_u_NH = self.act_func(right_y_NH) + (right_u_NH if right_u_NH is not None else 0.0)
            if self.post_norm:
                next_layer_right_u_NH = self.norm(next_layer_right_u_NH)
        
        return right_x_NP, next_layer_left_u_NH, next_layer_right_u_NH
    
    def _ssm(
        self,
        left_u_NH: Optional[torch.Tensor],
        right_u_NH: Optional[torch.Tensor],
        tilde_Eα_NP: torch.Tensor,
        dt_N: torch.Tensor,
        x_0_P: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        *leading_dims, N, P = tilde_Eα_NP.shape
        
        Λ_i = self.get_Λ_i(right_u_NH, shift_u=True)
        if "Λ_i_P" in Λ_i: # we use this key check instead of shape check because both cases end in the dimension of P
            # TODO: check why there is no ellipsis in second argument of einsum. no batch?
            Λ_dt_NP = torch.einsum("...n,p->...np", dt_N, Λ_i["Λ_i_P"]) # element-wise product but a diagonal matrix
        else:
            # scaling each row of Λ_i_NP by associated element of dt_N
            Λ_dt_NP = torch.einsum("...n,...np->...np", dt_N, Λ_i["Λ_i_NP"]) 
        
        # compute left-hand Du    
        if left_u_NH is not None:
            left_Du_NH = torch.einsum("...nh,h->...nh", left_u_NH, self.D_H)
        else:
            assert self.is_first_layer
            left_Du_NH = 0.0
            
        # computing right-hand Bu and Du
        if right_u_NH is not None:
            right_u_NH = F.pad(right_u_NH[..., :-1, :], (0, 0, 1, 0))
            right_Bu_NP = torch.einsum( # Equation 19 of S2P2 paper
                "...np,ph,...nh->...np",
                Λ_dt_NP.exp() - 1, # Λ_bar - 1
                self.B_tilde_PH,
                right_u_NH.type(torch.complex64)
            )
            right_Du_NH = torch.einsum("...nh,h->...nh", right_u_NH, self.D_H)
        else:
            assert self.is_first_layer
            right_Du_NH = right_Bu_NP =  0.0
            
        # parallel scan as per (Heinsen 2023) and https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
        log_impulse_Np1_P = torch.concat((x_0_P.unsqueeze(-2), right_Bu_NP + tilde_Eα_NP), dim = -2).log()
        Λ_dt_star = F.pad(Λ_dt_NP.cumsum(-2), (0, 0, 1, 0))
        right_log_x_NP = torch.logcumsumexp(log_impulse_Np1_P - Λ_dt_star, -2) + Λ_dt_star
        right_x_NP = right_log_x_NP.exp() # Contains initial_state_P in index 0
        left_x_NP = Λ_dt_NP.exp() * right_x_NP[..., :-1, :] + right_Bu_NP # this is Equation (15)
        right_x_NP = right_x_NP[..., 1:, :] # finally we exponentiate, ignoring the first dimension padded earlier
        
        left_y_NH = 2 * torch.einsum("hp,...np->...nh", self.C_tilde_HP, left_x_NP).real + left_Du_NH
        right_y_NH = 2 * torch.einsum("hp,...np->...nh", self.C_tilde_HP, right_x_NP).real + right_Du_NH
        
        return right_x_NP, left_y_NH, right_y_NH
        
    def get_x_left_limit(
        self,
        right_x_P: torch.Tensor,
        dt_G: torch.Tensor,
        current_right_u_H: torch.Tensor
    ) -> torch.Tensor:
        
        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)
            
        Λ_i = self.get_Λ_i(current_right_u_H, shift_u = False) # input signal u should already been shifted
        if "Λ_i_P" in Λ_i:
            Λ_bar_GP = torch.exp(torch.einsum("...g,p->...gp", dt_G, Λ_i["Λ_i_P"]))
        else:
            Λ_bar_GP = torch.exp(torch.einsum("...g,...p->...gp", dt_G, Λ_i["Λ_i_NP"]))
            
        Λ_bar_x_GP = torch.einsum("...p,...gp->...gp", right_x_P, Λ_bar_GP)
        
        if current_right_u_H is None:
            assert self.is_first_layer
            return Λ_bar_x_GP
        else: # add Bu to impulse
            if self.pre_norm:
                current_right_u_H = self.norm(current_right_u_H)
            impulse_GP = torch.einsum(
                "...gp,ph,...h->...gp", 
                Λ_bar_GP - 1.0, 
                self.B_tilde_PH, 
                current_right_u_H.type(torch.complex64))
            return Λ_bar_x_GP + impulse_GP
    
    def depth_pass(
        self,
        current_left_x_P: torch.Tensor,
        current_left_u_H: Optional[torch.Tensor]
    ) -> torch.Tensor:
        
        if current_left_u_H is not None:
            if self.pre_norm:
                normed_u_H = self.norm(current_left_u_H)
            else:
                normed_u_H = current_left_u_H
            left_Du_H = torch.einsum("...h,h->...h", normed_u_H, self.D_H)
        else:
            assert self.is_first_layer
            left_Du_H = 0.0
            
        y_H = 2 * torch.einsum("...p,hp->...h", current_left_x_P, self.C_tilde_HP).real + left_Du_H
        
        if self.post_norm:
            new_u_H = self.norm(self.act_func(y_H) + (current_left_u_H if current_left_u_H is not None else 0.0))
        else:
            new_u_H = self.act_func(y_H) + (current_left_u_H if current_left_u_H is not None else 0.0)
            
        return new_u_H
        