"""Round-robin decode with bounded host, index and device admission."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .config import CapacityError
from .model import Model


@dataclass
class Request:
    request_id: str
    prompt: list[int]
    max_new_tokens: int
    generated: list[int] = field(default_factory=list)
    prefill_count: int = 0
    decode_count: int = 0
    finished: bool = False
    cancelled: bool = False


class Engine:
    def __init__(
        self,
        model: Model,
        cache,
        top_k=None,
        backend="auto",
        eos_token_id: int | None = None,
        selector: str = "quest",
    ):
        cfg, kv = model.config, cache.config
        if (cfg.num_key_value_heads, cfg.head_dim) != (kv.kv_heads, kv.head_dim):
            raise ValueError("model/cache head dimensions differ")
        if model.device != cache.pool.device or model.dtype != cache.pool.dtype:
            raise ValueError("model/cache dtype or device differ")
        if top_k is not None and top_k < 1:
            raise ValueError("top_k must be positive or None")
        self.model, self.cache, self.top_k, self.backend = model, cache, top_k, backend
        self.eos_token_id = eos_token_id
        self.selector = selector
        self.requests: dict[str, Request] = {}
        self.waiting: deque[Request] = deque()
        self.running: deque[Request] = deque()

    def add_request(self, request_id, prompt, max_new_tokens=16):
        if request_id in self.requests or not prompt or max_new_tokens < 1:
            raise ValueError("need a unique ID, nonempty prompt, and positive output length")
        if any(token < 0 or token >= self.model.config.vocab_size for token in prompt):
            raise ValueError("token outside model vocabulary")
        needed = -(-(len(prompt) + max_new_tokens) // self.cache.config.page_size)
        if needed * self.model.config.num_hidden_layers > self.cache.config.host_pages:
            raise CapacityError("request cannot fit even in an empty host pool")
        if self.cache.admission_blocks(self.model.config.num_hidden_layers) > self.cache.budget:
            raise CapacityError("request hot-region + tail footprint cannot fit the device pool")
        index_bytes = self.cache.index_reservation_bytes(
            len(prompt) + max_new_tokens, self.model.config.num_hidden_layers
        )
        if index_bytes > self.cache.config.index_budget_bytes:
            raise CapacityError("request index cannot fit even in an empty index budget")
        request = Request(request_id, list(prompt), max_new_tokens)
        self.requests[request_id] = request
        self.waiting.append(request)
        return request

    def _emit(self, request, logits):
        token = int(logits.argmax().item())
        request.generated.append(token)
        if token == self.eos_token_id or len(request.generated) >= request.max_new_tokens:
            request.finished = True
            self.cache.release_request(request.request_id, wait=False)
        return token

    def step(self):
        """Admit at most one prompt and give EVERY running request a decode turn.

        This is a sequential scheduler, not a batched model runner.
        A long waiting queue cannot starve the requests already decoding.
        """
        emitted = []
        self.cache.collect()
        if self.waiting:
            request = self.waiting[0]
            admitted = False
            try:
                self.cache.reserve(
                    request.request_id,
                    len(request.prompt) + request.max_new_tokens,
                    self.model.config.num_hidden_layers,
                )
                admitted = True
            except CapacityError:
                if not self.running:
                    if not self.cache.storage.retired:
                        raise
                    self.cache.collect(wait=True)
                    self.cache.reserve(
                        request.request_id,
                        len(request.prompt) + request.max_new_tokens,
                        self.model.config.num_hidden_layers,
                    )
                    admitted = True
            if admitted:
                self.waiting.popleft()
                logits = self.model.prefill(request.prompt, self.cache, request.request_id)
                request.prefill_count += 1
                emitted.append((request.request_id, self._emit(request, logits)))
                if not request.finished:
                    self.running.append(request)
        for _ in range(len(self.running)):
            request = self.running.popleft()
            logits = self.model.decode(
                request.generated[-1],
                request.request_id,
                self.cache,
                self.top_k,
                self.backend,
                selector=self.selector,
            )
            request.decode_count += 1
            emitted.append((request.request_id, self._emit(request, logits)))
            if not request.finished:
                self.running.append(request)
        return emitted

    def run(self):
        while self.waiting or self.running:
            self.step()
        self.cache.collect(wait=True)
        return {name: request.generated for name, request in self.requests.items()}

    def cancel(self, request_id):
        request = self.requests[request_id]
        if request.finished:
            return
        if request in self.waiting:
            self.waiting.remove(request)
        else:
            self.running.remove(request)
            self.cache.release_request(request_id, wait=False)
        request.finished = request.cancelled = True
