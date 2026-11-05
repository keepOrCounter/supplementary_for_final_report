from __future__ import annotations
import warnings
import instructor
from dataclasses import dataclass
from pydantic import BaseModel
from openai import OpenAI

from core.schemas import ActionChoice, bounded_action_choice


@dataclass
class LLMCallStats:
    """Stats from a single LLM call (final successful completion only)."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    seed_used: int | None = None


@dataclass
class LLMCumulativeStats:
    """Cumulative stats across all LLM calls in one client instance.

    Token counts include ALL API calls (including failed retry attempts),
    tracked via instructor hooks on completion:response.
    """
    total_calls: int = 0 # number of LLM call (chat_structured invocations)
    retry_count: int = 0 # answers instructor refused; a request was sent again for each,
    # so total_calls + retry_count is the number of API requests (one high for a call
    # that died, whose last refusal is counted but is not sent again)
    total_prompt_tokens: int = 0 # prompt tokens from ALL API requests (including retries)
    total_completion_tokens: int = 0
    total_tokens: int = 0
    token_tracking_complete: bool = True # False if any API response lacked usage data

    def to_dict(self) -> dict:
        return {
            "total_calls": self.total_calls,
            "retry_count": self.retry_count,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "token_tracking_complete": self.token_tracking_complete,
        }


class LLMClient:
    """OpenAI-compatible LLM client with structured output"""

    def __init__(self, model: str, base_url: str, api_key: str = "ollama", max_retries: int = 3, \
                    temperature: float = 0.7, top_p: float = 0.9, max_tokens: int = -1, \
                        reasoning_effort: str = "high",):
        self.model = model
        self.max_retries = max_retries
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens # caps output length via /v1 (= Ollama num_predict; -1 = unlimited)
        self.reasoning_effort = reasoning_effort # think: "none" = off, "low"/"medium"/"high" = on

        self.client = OpenAI(base_url=base_url, api_key=api_key) # basic client


        # Per-attempt seed rotation: instructor reuses the SAME seed across retries, so a bad output is reproduced
        # ("wrong once, wrong forever"). Here rotate the seed per attempt instead in _build_instructor_client.
        self._attempt_index: int = 0 # underlying API calls within the CURRENT chat_structured
        self.instructor_client = self._build_instructor_client()

        # LLM Stats tracking
        self.stats = LLMCumulativeStats()
        self.last_call_stats: LLMCallStats = None

        # Register hooks to track ALL API responses (including retries)
        self.instructor_client.on("completion:response", self.__on_completion_response)
        self.instructor_client.on("completion:error", self.__on_completion_error)
        self.instructor_client.on("parse:error", self.__on_completion_error)

    def _build_instructor_client(self) -> instructor.Instructor:
        """Build the instructor client with per-attempt seed rotation baked in.
        
        attempt 0 -> seed,  attempt 1 -> seed+1,  attempt 2 -> seed+2, ...

        Inject the wrapper as the create instructor actually loops on
        (via instructor.patch(create=...)); replacing client attributes after
        from_openai() has no effect because instructor captures the original
        reference inside its retry closure.
        """
        raw_create = self.client.chat.completions.create

        def create_with_rotating_seed(*args, **kwargs):
            base_seed = kwargs.get("seed")
            if base_seed is not None:
                kwargs["seed"] = base_seed + self._attempt_index
            self._attempt_index += 1
            return raw_create(*args, **kwargs)

        patched = instructor.patch(create=create_with_rotating_seed, mode=instructor.Mode.TOOLS)
        return instructor.Instructor(client=self.client, create=patched, mode=instructor.Mode.TOOLS)

    def __on_completion_response(self, response) -> None:
        """Hook: triggered on EVERY raw API response (including retry attempts).
        Tracks token usage from all attempts, not just the final successful one.
        """
        try:
            usage = response.usage
            if usage:
                self.stats.total_prompt_tokens += usage.prompt_tokens
                self.stats.total_completion_tokens += usage.completion_tokens
                self.stats.total_tokens += usage.total_tokens
            else:
                self.stats.token_tracking_complete = False # flag under-counted totals into the data
                warnings.warn("LLM API response has no usage data: token tracking will be incomplete.", \
                    RuntimeWarning)
        except AttributeError:
            self.stats.token_tracking_complete = False # flag under-counted totals into the data
            warnings.warn(
                "LLM API response missing 'usage' attribute(or usage contains 'None' type attribute): token tracking will be incomplete.", \
                RuntimeWarning)

    def __on_completion_error(self, error, **kwargs) -> None:
        """Hook: triggered when instructor needs to retry.

        Listens to BOTH completion:error (API/network errors) and parse:error
        (LLM output validation failures).
        """
        self.stats.retry_count += 1

    def close(self) -> None:
        """release the HTTP connections this client holds

        A runner that builds one client per agent per game leaves one connection
        pool open per client, so it closes them when the game is over. The stats
        stay readable afterwards; only the transport goes.
        """
        self.client.close()

    def chat_basic(self, user_prompt: str, system_prompt: str = None) -> str:
        """Basic LLM call, return raw response"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self.client.chat.completions.create(model=self.model, messages=messages, \
            temperature=self.temperature, top_p=self.top_p, max_tokens=self.max_tokens, \
                reasoning_effort=self.reasoning_effort)

        return response.choices[0].message.content

    def chat_structured(self, user_prompt: str, system_prompt: str = None, max_index: int = None, \
        seed: int = None, reasoning_effort: str = None) -> tuple[ActionChoice, str]:
        """structured LLM call using instructor, returns class with specified format

        Args:
            max_index: if set, validates action_index into [0, max_index]
            seed: optional seed for reproducible output (passed to LLM API)

        Returns:
            tuple of (ActionChoice, reasoning_text)
            reasoning_text is the model's thinking/reasoning content (None if not available)

        Side effects:
            Updates self.stats (cumulative via hooks) and self.last_call_stats (per-call).
        """
        # Action-choice schema selection; the actual call is shared with
        # chat_structured_model to avoid duplicating the instructor/stats plumbing.
        response_model = bounded_action_choice(max_index) if max_index is not None else ActionChoice
        return self.chat_structured_model(user_prompt, response_model, \
            system_prompt=system_prompt, seed=seed, reasoning_effort=reasoning_effort)

    def chat_structured_model(self, user_prompt: str, response_model: type[BaseModel], \
        system_prompt: str = None, seed: int = None, reasoning_effort: str = None) -> tuple[BaseModel, str]:
        """Canonical structured LLM call for an ARBITRARY pydantic response_model.

        Shared machinery (instructor + retries + token hooks + reasoning extraction)
        behind both this method and chat_structured. Reusable beyond action choices
        (e.g. the soft-likelihood judge).

        ``reasoning_effort`` overrides the client default for THIS call only (e.g.
        "none" to disable the model's thinking); None keeps the configured value.

        Returns:
            tuple of (response_model instance, reasoning_text or None)

        Side effects:
            Updates self.stats (cumulative via hooks) and self.last_call_stats (per-call).
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        self.stats.total_calls += 1
        self._attempt_index = 0  # reset per-attempt seed rotation for this call

        try:
            result, raw_completion = self.instructor_client.chat.completions.create_with_completion(model=self.model, \
                response_model=response_model, max_retries=self.max_retries, messages=messages, \
                    temperature=self.temperature, top_p=self.top_p, seed=seed, max_tokens=self.max_tokens, \
                        reasoning_effort=reasoning_effort if reasoning_effort is not None else self.reasoning_effort)
        except Exception as e:
            self.last_call_stats = None
            raise e

        # Per-call token usage (cumulative tracking is handled by the completion hook,
        # which captures all attempts including retries).
        call_stats = LLMCallStats(seed_used=seed)
        try:
            usage = raw_completion.usage
            if usage:
                call_stats.prompt_tokens = usage.prompt_tokens
                call_stats.completion_tokens = usage.completion_tokens
                call_stats.total_tokens = usage.total_tokens
            else:
                warnings.warn("LLM completion has no usage data: per-call token stats unavailable.", \
                    RuntimeWarning)
        except AttributeError:
            warnings.warn("LLM completion missing 'usage' attribute: per-call token stats unavailable.", \
                RuntimeWarning)

        self.last_call_stats = call_stats

        # Extract reasoning (thinking) content if available
        reasoning = None
        try:
            reasoning = raw_completion.choices[0].message.reasoning
        except AttributeError:
            pass  # model does not support reasoning/thinking: this is normal
        except IndexError:
            raise ValueError("LLM completion has empty choices list: cannot extract response.")

        return result, reasoning


def create_llm_client(config: dict) -> LLMClient:
    """create LLMClient from config dict

    NOTE: Ollama's OpenAI-compatible /v1 endpoint ignores extra_body options
    (num_ctx, num_predict, repeat_penalty, keep_alive). So:
        - output length -> max_tokens (real /v1 param)
        - thinking on/off -> reasoning_effort (real /v1 param)
        - num_ctx / repeat_penalty -> bake into a model variant via Modelfile
            (see Modelfile.template) and set `model:` to that variant
        - keep_alive -> OLLAMA_KEEP_ALIVE env var when launching `ollama serve`
    """
    llm_config = config["llm"]

    return LLMClient(model=llm_config["model"], base_url=llm_config["base_url"], \
        api_key=llm_config.get("api_key", "ollama"), max_retries=llm_config.get("max_retries", 3),\
            temperature=llm_config.get("temperature", 0.7), top_p=llm_config.get("top_p", 0.9), \
                max_tokens=llm_config.get("max_tokens", -1), reasoning_effort=llm_config.get("reasoning_effort", "high"))