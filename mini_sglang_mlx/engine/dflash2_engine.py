"""DFlash2 speculative-decoding engine.

Same block-diffusion skeleton as :class:`DflashEngine` — one draft
forward proposes ``K = block_size - 1`` tokens, target verify accepts
the longest matching prefix — with the draft's own proposal step
replaced by DFlash2's candidate selector.

DFlash's ``argmax`` per masked slot picks each token in isolation, so
the block can be individually plausible yet incoherent, and verify
then truncates it.  DFlash2 instead keeps the top ``selector_top_k``
candidates per slot and walks the best *path* through them, scoring
each adjacent pair with the draft's own logit plus a low-rank
bilinear term over a context-gated hidden state (see
:class:`mini_sglang_mlx.models.dflash2.CandidateSelector`).  The walk is the
only sequential part; scoring is fully parallel.

Everything else is inherited verbatim, including the calibration
invariant that keeps the draft KV cache in lockstep with the target
at every iter boundary — DFlash2 changes how drafts are *chosen*, not
which K/V the draft reads.
"""

from __future__ import annotations

import mlx.core as mx

from mini_sglang_mlx.core import Batch, use_ctx

from .dflash_engine import DflashEngine


class Dflash2Engine(DflashEngine):
    """DFlash2: block diffusion + candidate-selector path walk."""

    spec_algo = "dflash2"
    draft_model_type = "dflash2"

    def _propose_drafts(self, proposal_batch: Batch) -> mx.array:
        """Propose via the draft's selector walk instead of per-slot argmax.

        :meth:`mini_sglang_mlx.models.dflash2.Model.propose` runs the block
        forward, scores every adjacent ``(predecessor, candidate)``
        pair over the top-k shortlist and returns the best path — one
        token per masked slot, in the same ``[B, K]`` layout the
        verify path expects (only the sampler differs; the greedy-only
        walk matches the greedy-only verification).
        """
        with (
            use_ctx(self.draft_ctx),
            self.draft_ctx.forward_batch(proposal_batch),
        ):
            return self.draft_model.propose(proposal_batch.input_ids)
