"""KDA kernel backed by the gfx950 TLX kernels.

Subclasses ``TritonKDAKernel`` rather than wrapping it: TLX only covers the
decode recurrence and the chunk prefill, so everything else -- target_verify,
packed decode, ReplaySSM, intermediate-state extends -- should behave exactly
as Triton does, and inheriting gets that for free (including the dispatcher's
``isinstance`` safe-gate check, which the TLX path does honor: see below).

The TLX kernels take prepared inputs (l2-normalized Q/K, a materialized per-K
log decay, a sigmoided beta) while the dispatcher hands every kernel the raw
conv output plus ``A_log`` / ``dt_bias`` / ``lower_bound``. ``prepare_kda_inputs``
bridges that in one Triton launch, and is also where KDA's safe gate is
applied, since the TLX kernels have no ``lower_bound`` path of their own.
"""

from typing import Optional

import torch

from sglang.kernels.ops.kimi_k3 import tlx as tlx_kernels
from sglang.kernels.ops.kimi_k3.tlx.kimi_k3_kda_prepare import prepare_kda_inputs
from sglang.srt.layers.attention.linear.kernels.kda_triton import TritonKDAKernel
from sglang.srt.utils.common import rank0_log


class TlxKDAKernel(TritonKDAKernel):
    """Decode and extend on TLX, every other phase inherited from Triton."""

    # The TLX decode consumes split Q/K/V, so route batched decode through
    # decode() instead of the packed path -- otherwise only models with a
    # safe gate (which skip packed decode anyway) would ever reach TLX.
    supports_packed_decode: bool = False

    def __init__(self):
        if not tlx_kernels.is_available():
            raise ValueError(
                "KDA TLX backend needs a TLX-capable triton; launch under "
                "PYTHONPATH=$SGLANG_TLX_PREFIX (image built with BUILD_TLX=1)."
            )
        super().__init__()
        # Probed once: the prefill kernel compiles against a newer TLX than
        # the decode one, so a wheel can support decode and not prefill.
        self._prefill_ok = tlx_kernels.is_prefill_available()
        if not self._prefill_ok:
            rank0_log(
                "KDA TLX prefill disabled: this triton is missing "
                f"{', '.join(tlx_kernels.missing_prefill_ops())}. "
                "Decode uses TLX, extend stays on Triton."
            )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        lower_bound: Optional[float] = None,
        **kwargs,
    ) -> torch.Tensor:
        if kwargs.get("replayssm_d") is not None:
            return super().decode(
                q, k, v, a, b, A_log=A_log, dt_bias=dt_bias, ssm_states=ssm_states,
                cache_indices=cache_indices, query_start_loc=query_start_loc,
                lower_bound=lower_bound, **kwargs,
            )

        HV, V, K = ssm_states.shape[-3:]
        q_n, k_n, g, beta = prepare_kda_inputs(
            q, k, a, b, A_log, dt_bias, num_heads=HV, head_dim=K,
            lower_bound=lower_bound, sigmoid_beta=True,
        )
        return tlx_kernels.kda_recurrent_decode(
            q_n,
            k_n,
            v.contiguous(),
            g,
            beta,
            scale=K**-0.5,
            state_pool=ssm_states,
            # One pool, updated in place: the TLX kernel's split read/write
            # indices exist for checkpointed state, which KDA does not use here.
            read_indices=cache_indices,
            write_indices=cache_indices,
            cu_seqlens=query_start_loc,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        A_log: Optional[torch.Tensor] = None,
        dt_bias: Optional[torch.Tensor] = None,
        lower_bound: Optional[float] = None,
        beta_is_raw: bool = False,
        return_intermediate_states: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            not self._prefill_ok
            or return_intermediate_states
            or kwargs.get("track_state") is not None
            or A_log is None
        ):
            return super().extend(
                q, k, v, g, beta, ssm_states=ssm_states, cache_indices=cache_indices,
                query_start_loc=query_start_loc, A_log=A_log, dt_bias=dt_bias,
                lower_bound=lower_bound, beta_is_raw=beta_is_raw,
                return_intermediate_states=return_intermediate_states, **kwargs,
            )

        HV, V, K = ssm_states.shape[-3:]
        q_n, k_n, log_g, beta_act = prepare_kda_inputs(
            q, k, g, beta, A_log, dt_bias, num_heads=HV, head_dim=K,
            lower_bound=lower_bound, sigmoid_beta=beta_is_raw,
        )

        # kda_paged_prefill takes a dense [N, H, V, K] block per sequence and
        # returns a new one, so the indexed pool is gathered in and written back.
        initial_state = ssm_states[cache_indices.long()].contiguous()
        out, final_state = tlx_kernels.kda_paged_prefill(
            q_n,
            k_n,
            v.contiguous(),
            log_g,
            beta_act,
            scale=K**-0.5,
            initial_state=initial_state,
            cu_seqlens=query_start_loc,
        )
        ssm_states[cache_indices.long()] = final_state
        return out, None
