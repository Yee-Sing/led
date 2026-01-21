from abc import abstractmethod
from typing import Callable
import torch
import torch.nn as nn
from spikingjelly.activation_based.neuron import surrogate, base
from spikingjelly.activation_based import configure
import math
import numpy as np
import logging
from .auto_cuda import neuron_kernel as ac_neuron_kernel

try:
    import cupy
    from spikingjelly.activation_based.neuron import neuron_kernel, cuda_utils

except BaseException as e:
    logging.info(f'spikingjelly.activation_based.neuron: {e}')
    cupy = None
    neuron_kernel = None
    cuda_utils = None


class BaseNode(base.MemoryModule):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False,
                 step_mode='s', backend='torch', store_v_seq: bool = False):

        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory('v', 0.)
        else:
            self.register_memory('v', v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

        # used in lava_exchange
        self.lava_s_cale = 1 << 6

        # used for cupy backend
        self.forward_kernel = None
        self.backward_kernel = None

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value:
            if not hasattr(self, 'v_seq'):
                self.register_memory('v_seq', None)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1. - spike) * v + spike * v_reset
        return v

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):

        raise NotImplementedError

    def neuronal_fire(self):


        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

        else:
            # hard reset
            self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}'

    def single_step_forward(self, x: torch.Tensor):

        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)

        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)

        return torch.stack(y_seq)

    def v_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.v, float):
            v_init = self.v
            self.v = torch.full_like(x.data, v_init)


class AdaptBaseNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 v_rest: float = 0., w_rest: float = 0., tau_w: float = 2., a: float = 0., b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        # b: jump amplitudes
        # a: subthreshold coupling
        assert isinstance(w_rest, float)
        assert isinstance(v_rest, float)
        assert isinstance(tau_w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.register_memory('w', w_rest)

        self.w_rest = w_rest
        self.v_rest = v_rest
        self.tau_w = tau_w
        self.a = a
        self.b = b

    @staticmethod
    @torch.jit.script
    def jit_neuronal_adaptation(w: torch.Tensor, tau_w: float, a: float, v_rest: float, v: torch.Tensor):
        return w + 1. / tau_w * (a * (v - v_rest) - w)

    def neuronal_adaptation(self):

        self.w = self.jit_neuronal_adaptation(self.w, self.tau_w, self.a, self.v_rest, self.v)

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(v: torch.Tensor, w: torch.Tensor, spike_d: torch.Tensor, v_reset: float, b: float,
                       spike: torch.Tensor):
        v = (1. - spike_d) * v + spike * v_reset
        w = w + b * spike
        return v, w

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(v: torch.Tensor, w: torch.Tensor, spike_d: torch.Tensor, v_threshold: float, b: float,
                       spike: torch.Tensor):
        v = v - spike_d * v_threshold
        w = w + b * spike
        return v, w

    def neuronal_reset(self, spike):

        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v, self.w = self.jit_soft_reset(self.v, self.w, spike_d, self.v_threshold, self.b, spike)

        else:
            # hard reset
            self.v, self.w = self.jit_hard_reset(self.v, self.w, spike_d, self.v_reset, self.b, spike)

    def extra_repr(self):
        return super().extra_repr() + f', v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}'

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.w_float_to_tensor(x)
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            w_init = self.w
            self.w = torch.full_like(x.data, fill_value=w_init)


class IFNode(BaseNode):
    def __init__(self, v_threshold: float = 1., v_reset: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(x: torch.Tensor, v: torch.Tensor, v_threshold: float):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                               v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                          v_reset: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
            elif self.backend == 'cupy':
                hard_reset = self.v_reset is not None

                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype):

                    self.forward_kernel = ac_neuron_kernel.IFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype):

                    self.backward_kernel = ac_neuron_kernel.IFNodeBPTTKernel(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = ac_neuron_kernel.IFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0),
                                                                    self.v_threshold, self.v_reset, self.forward_kernel,
                                                                    self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()

                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_with_v_seq(x_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(x_seq, self.v, self.v_threshold)
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_with_v_seq(x_seq,
                                                                                                           self.v,
                                                                                                           self.v_threshold,
                                                                                                           self.v_reset)
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(x_seq, self.v, self.v_threshold,
                                                                                    self.v_reset)
            return spike_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(x, self.v, self.v_threshold)
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(x, self.v, self.v_threshold, self.v_reset)
            return spike


class LIFNode(BaseNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):

        assert isinstance(tau, float) and tau > 1.

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.tau = tau
        self.decay_input = decay_input

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(x, self.v, self.v_reset, self.tau)

        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(x, self.v, self.v_reset, self.tau)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v + (x - v) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input_reset0(x: torch.Tensor, v: torch.Tensor, tau: float):
        v = v * (1. - 1. / tau) + x
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            v_reset: float, tau: float):
        v = v + (x - (v - v_reset)) / tau
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               v_reset: float, tau: float):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1. - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                            tau: float):
        v = v + (x - v) / tau
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                               tau: float):
        v = v * (1. - 1. / tau) + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - (v - v_reset)) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              v_reset: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float, v_reset: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v - (v - v_reset) / tau + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1. - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                           tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                      v_threshold: float, tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + (x_seq[t] - v) / tau
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float,
                                                              tau: float):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(x_seq: torch.Tensor, v: torch.Tensor,
                                                                         v_threshold: float,
                                                                         tau: float):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v * (1. - 1. / tau) + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            return super().single_step_forward(x)
        else:
            self.v_float_to_tensor(x)
            if self.v_reset is None:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_decay_input(x, self.v,
                                                                                             self.v_threshold, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_soft_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.tau)
            else:
                if self.decay_input:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_decay_input(x, self.v,
                                                                                             self.v_threshold,
                                                                                             self.v_reset, self.tau)
                else:
                    spike, self.v = self.jit_eval_single_step_forward_hard_reset_no_decay_input(x, self.v,
                                                                                                self.v_threshold,
                                                                                                self.v_reset,
                                                                                                self.tau)
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.training:
            if self.backend == 'torch':
                return super().multi_step_forward(x_seq)
            elif self.backend == 'cupy':

                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = 'float'
                elif x_seq.dtype == torch.half:
                    dtype = 'half2'
                else:
                    raise NotImplementedError(x_seq.dtype)

                if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input):
                    self.forward_kernel = ac_neuron_kernel.LIFNodeFPTTKernel(decay_input=self.decay_input, hard_reset=hard_reset, dtype=dtype)

                if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                        detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                    self.backward_kernel = ac_neuron_kernel.LIFNodeBPTTKernel(decay_input=self.decay_input, surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)

                self.v_float_to_tensor(x_seq[0])

                spike_seq, v_seq = ac_neuron_kernel.LIFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(0),
                                                                     self.v_threshold, self.v_reset, 1. / self.tau,
                                                                     self.forward_kernel,
                                                                     self.backward_kernel)

                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)

                if self.store_v_seq:
                    self.v_seq = v_seq

                self.v = v_seq[-1].clone()

                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])
            if self.v_reset is None:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_decay_input(x_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_soft_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset_no_decay_input(x_seq, self.v,
                                                                                                       self.v_threshold,
                                                                                                       self.tau)
            else:
                if self.decay_input:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_decay_input(x_seq, self.v,
                                                                                                    self.v_threshold,
                                                                                                    self.v_reset,
                                                                                                    self.tau)
                else:
                    if self.store_v_seq:
                        spike_seq, self.v, self.v_seq = self.jit_eval_multi_step_forward_hard_reset_no_decay_input_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset, self.tau)
                    else:
                        spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset_no_decay_input(x_seq, self.v,
                                                                                                       self.v_threshold,
                                                                                                       self.v_reset,
                                                                                                       self.tau)

            return spike_seq


class ParametricLIFNode(BaseNode):
    def __init__(self, init_tau: float = 2.0, decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):

        assert isinstance(init_tau, float) and init_tau > 1.
        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.decay_input = decay_input
        init_w = - math.log(init_tau - 1.)
        self.w = nn.Parameter(torch.as_tensor(init_w))

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1. / self.w.sigmoid()
        return super().extra_repr() + f', tau={tau}'

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.:
                self.v = self.v * (1. - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            hard_reset = self.v_reset is not None
            if x_seq.dtype == torch.float:
                dtype = 'float'
            elif x_seq.dtype == torch.half:
                dtype = 'half2'
            else:
                raise NotImplementedError(x_seq.dtype)

            if self.forward_kernel is None or not self.forward_kernel.check_attributes(hard_reset=hard_reset,
                                                                                       dtype=dtype,
                                                                                       decay_input=self.decay_input):
                self.forward_kernel = ac_neuron_kernel.ParametricLIFNodeFPTTKernel(decay_input=self.decay_input,
                                                                         hard_reset=hard_reset, dtype=dtype)

            if self.backward_kernel is None or not self.backward_kernel.check_attributes(
                    surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset,
                    detach_reset=self.detach_reset, dtype=dtype, decay_input=self.decay_input):
                self.backward_kernel = ac_neuron_kernel.ParametricLIFNodeBPTTKernel(decay_input=self.decay_input,
                                                                          surrogate_function=self.surrogate_function.cuda_codes,
                                                                          hard_reset=hard_reset,
                                                                          detach_reset=self.detach_reset, dtype=dtype)


            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = ac_neuron_kernel.ParametricLIFNodeATGF.apply(
                x_seq.flatten(1), self.v.flatten(0), self.v_threshold, self.v_reset, self.w.sigmoid().to(x_seq),
                self.forward_kernel, self.backward_kernel)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class QIFNode(BaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1., v_rest: float = 0.,
                 v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):

        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert a0 > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepQIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.v_rest,
                self.v_c, self.a0, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class EIFNode(BaseNode):
    def __init__(self, tau: float = 2., delta_T: float = 1., theta_rh: float = .8, v_threshold: float = 1.,
                 v_rest: float = 0., v_reset: float = -0.1,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):


        assert isinstance(tau, float) and tau > 1.
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)
        self.tau = tau
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}'

    def neuronal_charge(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)

        self.v = self.v + (x + self.v_rest - self.v + self.delta_T * torch.exp(
            (self.v - self.theta_rh) / self.delta_T)) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])

            spike_seq, v_seq = neuron_kernel.MultiStepEIFNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.tau, self.v_threshold, self.v_reset, self.v_rest,
                self.theta_rh, self.delta_T, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class IzhikevichNode(AdaptBaseNode):
    def __init__(self, tau: float = 2., v_c: float = 0.8, a0: float = 1., v_threshold: float = 1.,
                 v_reset: float = 0., v_rest: float = -0.1, w_rest: float = 0., tau_w: float = 2., a: float = 0.,
                 b: float = 0.,
                 surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False, step_mode='s',
                 backend='torch', store_v_seq: bool = False):
        assert isinstance(tau, float) and tau > 1.
        assert a0 > 0

        super().__init__(v_threshold, v_reset, v_rest, w_rest, tau_w, a, b, surrogate_function, detach_reset, step_mode,
                         backend, store_v_seq)
        self.tau = tau
        self.v_c = v_c
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f', tau={self.tau}, v_c={self.v_c}, a0={self.a0}'

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c) - self.w) / self.tau

    @property
    def supported_backends(self):
        if self.step_mode == 's':
            return ('torch',)
        elif self.step_mode == 'm':
            return ('torch', 'cupy')
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == 'torch':
            return super().multi_step_forward(x_seq)
        elif self.backend == 'cupy':
            self.v_float_to_tensor(x_seq[0])
            self.w_float_to_tensor(x_seq[0])

            spike_seq, v_seq, w_seq = neuron_kernel.MultiStepIzhikevichNodePTT.apply(
                x_seq.flatten(1), self.v.flatten(0), self.w.flatten(0), self.tau, self.v_threshold, self.v_reset,
                self.v_rest, self.a, self.b, self.tau_w,
                self.v_c, self.a0, self.detach_reset, self.surrogate_function.cuda_code)

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            w_seq = w_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()
            self.w = w_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class LIAFNode(LIFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.act = act
        self.threshold_related = threshold_related

        assert self.backend == 'torch', "LIAFNode only supports for backend='torch'!"
        assert self.single_step_cupy_fp32_inference == False, "LIAFNode does not support for single_step_cupy_fp32_inference!"

    @property
    def supported_backends(self):
        return ('torch',)

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        if self.threshold_related:
            y = self.act(self.v - self.v_threshold)
        else:
            y = self.act(self.v)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return y


class KLIFNode(BaseNode):
    def __init__(self, scale_reset: bool=False, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, step_mode='s', backend='torch', store_v_seq: bool = False):

        assert isinstance(tau, float) and tau > 1.
        if backend == 'cupy':
            raise NotImplementedError("The CuPy backend for the KLIF neuron has not been implemented!")

        super().__init__(v_threshold, v_reset, surrogate_function, detach_reset, step_mode, backend, store_v_seq)

        self.scale_reset = scale_reset
        self.tau = tau
        self.decay_input = decay_input

        self.k = nn.Parameter(torch.as_tensor(1.))


    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, k: torch.Tensor):
        v = v + (x - (v - v_reset)) / tau
        v = torch.relu_(k * v)
        return v


    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, k: torch.Tensor):
        v = v - (v - v_reset) / tau + x
        v = torch.relu_(k * v)
        return v


    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            v_reset = 0.
        else:
            v_reset = self.v_reset
        if self.decay_input:
            self.v = self.neuronal_charge_decay_input(x, self.v, v_reset, self.tau, self.k)

        else:
            self.v = self.neuronal_charge_no_decay_input(x, self.v, v_reset, self.tau, self.k)


    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike


        if self.scale_reset:
            if self.v_reset is None:
                # soft reset
                self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold) / self.k

            else:
                # hard reset
                self.v = self.jit_hard_reset(self.v / self.k, spike_d, self.v_reset)

        else:

            if self.v_reset is None:
                # soft reset
                self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

            else:
                # hard reset
                self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)


