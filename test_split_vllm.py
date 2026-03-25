"""
test_split_vllm.py - Unit test suite for split_vllm.py
=======================================================

All vLLM engine calls, CUDA operations, and tokenizer I/O are mocked
so the entire suite runs on a standard CPU without any GPU or vLLM
installation required.

Run with:
    pytest test_split_vllm.py -v
"""

import argparse
import sys
import types
from io import StringIO
from unittest.mock import MagicMock, Mock, patch, call, mock_open

import pytest

# --------------------------------------------------------------------------- #
# vLLM and torch are not available in a CPU test environment.                 #
# Inject lightweight stubs into sys.modules BEFORE importing the module       #
# under test so that the top-level import block in split_vllm.py succeeds.   #
# --------------------------------------------------------------------------- #

def _make_vllm_stub() -> types.ModuleType:
    """Return a minimal vllm stub package."""
    vllm = types.ModuleType("vllm")

    # EngineArgs – only needs to be constructable.
    vllm.EngineArgs = MagicMock(name="EngineArgs")

    # LLMEngine – needs from_engine_args classmethod.
    mock_engine_cls = MagicMock(name="LLMEngine")
    mock_engine_cls.from_engine_args = MagicMock(
        return_value=MagicMock(name="llm_engine_instance")
    )
    vllm.LLMEngine = mock_engine_cls

    # SamplingParams – only needs to be constructable.
    vllm.SamplingParams = MagicMock(name="SamplingParams")

    # Sub-packages expected by the try/except import block.
    outputs_mod = types.ModuleType("vllm.outputs")
    outputs_mod.RequestOutput = MagicMock(name="RequestOutput")
    sys.modules["vllm.outputs"] = outputs_mod
    vllm.outputs = outputs_mod

    tokenizers_mod = types.ModuleType("vllm.tokenizers")
    get_tok_mod = types.ModuleType("vllm.tokenizers.get_tokenizer")
    get_tok_mod.get_tokenizer = MagicMock(name="get_tokenizer")
    sys.modules["vllm.tokenizers"] = tokenizers_mod
    sys.modules["vllm.tokenizers.get_tokenizer"] = get_tok_mod

    return vllm


def _make_torch_stub() -> types.ModuleType:
    """Return a minimal torch stub that exposes the CUDA helpers used."""
    torch = types.ModuleType("torch")
    cuda = types.ModuleType("torch.cuda")
    cuda.synchronize = MagicMock()
    cuda.empty_cache = MagicMock()

    nvtx = types.ModuleType("torch.cuda.nvtx")
    nvtx.range_push = MagicMock()
    nvtx.range_pop = MagicMock()
    cuda.nvtx = nvtx

    torch.cuda = cuda
    sys.modules["torch.cuda"] = cuda
    sys.modules["torch.cuda.nvtx"] = nvtx
    return torch


# Install stubs before the module is imported for the first time.
_vllm_stub = _make_vllm_stub()
_torch_stub = _make_torch_stub()
sys.modules.setdefault("vllm", _vllm_stub)
sys.modules.setdefault("torch", _torch_stub)

# Now we can safely import the module under test.
import split_vllm  # noqa: E402  (import after sys.modules patching)
from split_vllm import (  # noqa: E402
    EngineManager,
    InferencePipeline,
    PhaseStats,
    ProfileReporter,
    PromptBuilder,
    TextProcessor,
    build_arg_parser,
    LANGUAGE_MAP,
)


# =========================================================================== #
#  Shared helpers / fixtures                                                   #
# =========================================================================== #

def _make_tokenizer(
    eos_token: str = "<|endoftext|>",
    encode_return=None,
    decode_return: str = "clean text",
) -> MagicMock:
    """Return a mock tokenizer with configurable encode/decode behaviour."""
    tok = MagicMock(name="tokenizer")
    tok.eos_token = eos_token
    tok.encode.return_value = encode_return if encode_return is not None else [1, 2, 3]
    tok.decode.return_value = decode_return
    return tok


def _make_request_output(
    request_id: str = "inference",
    prompt_token_ids=None,
    gen_token_ids=None,
    text: str = "hello",
    finished: bool = True,
) -> MagicMock:
    """Build a mock RequestOutput-like object."""
    ro = MagicMock(name="RequestOutput")
    ro.request_id = request_id
    ro.prompt_token_ids = prompt_token_ids or [10, 20, 30]
    ro.finished = finished

    output = MagicMock()
    output.token_ids = gen_token_ids or [100, 200]
    output.text = text
    ro.outputs = [output]
    return ro


def _make_args(**kwargs) -> argparse.Namespace:
    """Return an argparse.Namespace with sane defaults, overridable via kwargs."""
    defaults = dict(
        input=None,
        text=None,
        output="out.txt",
        model="Qwen/Qwen2.5-1.5B",
        prompt=None,
        source=None,
        target=None,
        aggressive_normalize=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


# =========================================================================== #
#  PhaseStats                                                                  #
# =========================================================================== #

class TestPhaseStats:
    """PhaseStats is a frozen dataclass — test construction and immutability."""

    def test_construction_with_defaults(self):
        ps = PhaseStats("PREFILL", 0.5, "GPU")
        assert ps.phase_name == "PREFILL"
        assert ps.wall_s == 0.5
        assert ps.device == "GPU"
        assert ps.prompt_tokens == 0
        assert ps.gen_tokens == 0

    def test_construction_with_all_fields(self):
        ps = PhaseStats("DECODE", 1.23, "GPU", prompt_tokens=128, gen_tokens=64)
        assert ps.prompt_tokens == 128
        assert ps.gen_tokens == 64

    def test_immutability(self):
        ps = PhaseStats("PREFILL", 0.1, "GPU")
        with pytest.raises((AttributeError, TypeError)):
            ps.wall_s = 99.0  # type: ignore[misc]


# =========================================================================== #
#  TextProcessor.normalize                                                     #
# =========================================================================== #

class TestTextProcessorNormalize:
    """Happy paths and corner cases for the normalize static method."""

    def test_collapses_blank_lines(self):
        result = TextProcessor.normalize("line one\n\n\nline two")
        assert result == "line one\nline two"

    def test_strips_leading_trailing_whitespace_per_line(self):
        result = TextProcessor.normalize("  hello  \n  world  ")
        assert result == "hello\nworld"

    def test_non_aggressive_preserves_brackets_and_quotes(self):
        raw = '["hello", "world"]'
        result = TextProcessor.normalize(raw, aggressive=False)
        assert "[" in result or '"' in result  # structure preserved

    def test_aggressive_strips_brackets_and_quotes(self):
        raw = '["hello", "world"]'
        result = TextProcessor.normalize(raw, aggressive=True)
        assert "[" not in result
        assert "]" not in result
        assert '"' not in result

    def test_aggressive_strips_single_quotes_and_comma_joins(self):
        raw = "['alpha', 'beta']"
        result = TextProcessor.normalize(raw, aggressive=True)
        assert "'" not in result

    def test_empty_string_returns_empty(self):
        assert TextProcessor.normalize("") == ""

    def test_only_blank_lines_returns_empty(self):
        assert TextProcessor.normalize("\n\n\n") == ""

    def test_json_string_safe_without_aggressive(self):
        """JSON input must not be corrupted by default normalize."""
        json_str = '{"key": "value", "list": [1, 2, 3]}'
        result = TextProcessor.normalize(json_str, aggressive=False)
        assert '"key"' in result
        assert "[1, 2, 3]" in result

    def test_non_aggressive_default(self):
        """Calling normalize with one arg must default to non-aggressive."""
        raw = '["item"]'
        result = TextProcessor.normalize(raw)
        # Brackets should still be present in default mode.
        assert "[" in result


# =========================================================================== #
#  TextProcessor.clean_output                                                  #
# =========================================================================== #

class TestTextProcessorCleanOutput:
    """clean_output delegates to the tokenizer's encode/decode pipeline."""

    def test_returns_decoded_stripped_text(self):
        tok = _make_tokenizer(decode_return="  clean output  ")
        result = TextProcessor.clean_output("raw<|eos|>", tok)
        assert result == "clean output"

    def test_calls_encode_with_no_special_tokens(self):
        tok = _make_tokenizer()
        TextProcessor.clean_output("some text", tok)
        tok.encode.assert_called_once_with(
            "some text", add_special_tokens=False
        )

    def test_calls_decode_with_skip_special_tokens(self):
        tok = _make_tokenizer(encode_return=[7, 8, 9])
        TextProcessor.clean_output("some text", tok)
        tok.decode.assert_called_once_with(
            [7, 8, 9], skip_special_tokens=True
        )

    def test_empty_input(self):
        tok = _make_tokenizer(encode_return=[], decode_return="")
        result = TextProcessor.clean_output("", tok)
        assert result == ""

    def test_strips_trailing_whitespace_from_decode(self):
        tok = _make_tokenizer(decode_return="result\n  ")
        result = TextProcessor.clean_output("anything", tok)
        assert result == "result"


# =========================================================================== #
#  PromptBuilder.build                                                         #
# =========================================================================== #

class TestPromptBuilderBuild:
    """All routing paths through PromptBuilder.build."""

    # ---- Custom prompt paths ------------------------------------------- #

    def test_self_contained_prompt_no_input(self):
        args = _make_args(prompt="Describe AI.")
        result = PromptBuilder.build("", args)
        assert result == "Describe AI."

    def test_prompt_with_placeholder_injects_input(self):
        args = _make_args(prompt="Summarise:\n\n{input_text}")
        result = PromptBuilder.build("Hello world.", args)
        assert "Hello world." in result
        assert "{input_text}" not in result

    def test_prompt_without_placeholder_appends_input(self):
        args = _make_args(prompt="Do something with this:")
        result = PromptBuilder.build("my content", args)
        assert result.startswith("Do something with this:")
        assert "my content" in result
        assert result.endswith("Output:\n")

    def test_prompt_without_placeholder_structure(self):
        """Appended input should be separated by double newline."""
        args = _make_args(prompt="Prompt:")
        result = PromptBuilder.build("content", args)
        assert "Prompt:\n\ncontent\n\nOutput:\n" == result

    # ---- Translation fallback paths ------------------------------------ #

    def test_translation_fallback_uses_source_and_target(self):
        args = _make_args(source="en", target="hi")
        result = PromptBuilder.build("Hello", args)
        assert "English" in result
        assert "Hindi" in result
        assert "Hello" in result

    def test_translation_prompt_contains_rules(self):
        args = _make_args(source="en", target="fr")
        result = PromptBuilder.build("Some text.", args)
        assert "Do NOT summarize" in result
        assert "Translation:" in result

    def test_translation_unknown_lang_code_passthrough(self):
        """Unknown ISO codes should be used verbatim as language names."""
        args = _make_args(source="xx", target="yy")
        result = PromptBuilder.build("text", args)
        assert "xx" in result
        assert "yy" in result

    def test_translation_lang_codes_case_insensitive(self):
        args = _make_args(source="EN", target="HI")
        result = PromptBuilder.build("text", args)
        assert "English" in result
        assert "Hindi" in result

    # ---- Error path ----------------------------------------------------- #

    def test_raises_when_no_prompt_and_no_source_target(self):
        args = _make_args(prompt=None, source=None, target=None)
        with pytest.raises(ValueError, match="--source and --target are required"):
            PromptBuilder.build("text", args)

    def test_raises_when_no_prompt_and_only_source(self):
        args = _make_args(prompt=None, source="en", target=None)
        with pytest.raises(ValueError):
            PromptBuilder.build("text", args)

    def test_raises_when_no_prompt_and_only_target(self):
        args = _make_args(prompt=None, source=None, target="fr")
        with pytest.raises(ValueError):
            PromptBuilder.build("text", args)


# =========================================================================== #
#  LANGUAGE_MAP                                                                #
# =========================================================================== #

class TestLanguageMap:
    """Sanity checks on the constant — ensures no accidental regressions."""

    def test_required_codes_present(self):
        for code in ("en", "fr", "hi", "de", "es", "ja", "zh", "ar"):
            assert code in LANGUAGE_MAP, f"Missing language code: {code}"

    def test_values_are_non_empty_strings(self):
        for code, name in LANGUAGE_MAP.items():
            assert isinstance(name, str) and name, (
                f"Empty name for code: {code}"
            )


# =========================================================================== #
#  EngineManager._extract_data                                                 #
# =========================================================================== #

class TestEngineManagerExtractData:
    """Unit tests for the static helper that unpacks RequestOutput objects."""

    def test_happy_path(self):
        ro = _make_request_output(
            prompt_token_ids=[1, 2, 3],
            gen_token_ids=[10, 20],
            text="output text",
        )
        p, g, t = EngineManager._extract_data(ro)
        assert p == 3
        assert g == 2
        assert t == "output text"

    def test_no_prompt_token_ids(self):
        ro = _make_request_output()
        ro.prompt_token_ids = None
        p, g, t = EngineManager._extract_data(ro)
        assert p == 0

    def test_empty_outputs_list(self):
        ro = _make_request_output()
        ro.outputs = []
        p, g, t = EngineManager._extract_data(ro)
        assert g == 0
        assert t == ""

    def test_empty_prompt_tokens(self):
        ro = _make_request_output()
        ro.prompt_token_ids = []
        p, _, _ = EngineManager._extract_data(ro)
        # Empty list is falsy, so the branch takes the else path → 0.
        assert p == 0


# =========================================================================== #
#  EngineManager.run_phase                                                     #
# =========================================================================== #

class TestEngineManagerRunPhase:
    """
    Tests for the PREFILL / DECODE stepping logic.
    All engine I/O and CUDA calls are mocked.
    """

    def _make_manager(self) -> EngineManager:
        """Return an EngineManager whose inner engine is fully mocked."""
        with patch("split_vllm.EngineArgs"), \
             patch("split_vllm.LLMEngine") as mock_llm_cls:
            mock_llm_cls.from_engine_args.return_value = MagicMock(
                name="engine"
            )
            mgr = EngineManager(model="test-model")
        return mgr

    # ---- PREFILL -------------------------------------------------------- #

    def test_prefill_returns_on_first_generated_token(self):
        mgr = self._make_manager()
        ro = _make_request_output(
            request_id="inference",
            gen_token_ids=[99],
            text="first token",
            finished=False,
        )
        # Engine yields one unfinished step then stops.
        mgr.engine.has_unfinished_requests.side_effect = [True, False]
        mgr.engine.step.return_value = [ro]

        stats, text = mgr.run_phase("inference", "PREFILL")

        assert stats.phase_name == "PREFILL"
        assert stats.gen_tokens == 1
        assert stats.prompt_tokens == 3  # len([10, 20, 30])
        assert text == "first token"

    def test_prefill_ignores_outputs_for_other_request_ids(self):
        mgr = self._make_manager()
        other_ro = _make_request_output(
            request_id="other", gen_token_ids=[1], text="ignore me"
        )
        matching_ro = _make_request_output(
            request_id="inference", gen_token_ids=[2], text="mine"
        )
        mgr.engine.has_unfinished_requests.side_effect = [True, False]
        mgr.engine.step.return_value = [other_ro, matching_ro]

        stats, text = mgr.run_phase("inference", "PREFILL")
        assert text == "mine"

    # ---- DECODE --------------------------------------------------------- #

    def test_decode_runs_until_finished(self):
        mgr = self._make_manager()
        # Two steps: first unfinished, second finished.
        ro_step1 = _make_request_output(
            request_id="inference",
            gen_token_ids=[1],
            text="partial",
            finished=False,
        )
        ro_step2 = _make_request_output(
            request_id="inference",
            gen_token_ids=[1, 2, 3],
            text="full output",
            finished=True,
        )
        mgr.engine.has_unfinished_requests.side_effect = [True, True, False]
        mgr.engine.step.side_effect = [[ro_step1], [ro_step2]]

        stats, text = mgr.run_phase("inference", "DECODE")

        assert stats.phase_name == "DECODE"
        assert stats.device == "GPU"
        assert text == "full output"

    def test_decode_no_unfinished_requests_returns_zeroes(self):
        mgr = self._make_manager()
        mgr.engine.has_unfinished_requests.return_value = False

        stats, text = mgr.run_phase("inference", "DECODE")

        assert stats.prompt_tokens == 0
        assert stats.gen_tokens == 0
        assert text == ""

    def test_nvtx_range_pushed_and_popped(self):
        mgr = self._make_manager()
        mgr.engine.has_unfinished_requests.return_value = False

        mgr.run_phase("inference", "PREFILL")

        split_vllm.torch.cuda.nvtx.range_push.assert_called_with("PREFILL")
        split_vllm.torch.cuda.nvtx.range_pop.assert_called()

    def test_nvtx_range_pop_called_even_on_early_prefill_return(self):
        """range_pop must be called via the finally block in PREFILL."""
        mgr = self._make_manager()
        ro = _make_request_output(
            gen_token_ids=[1], text="t", finished=False
        )
        mgr.engine.has_unfinished_requests.side_effect = [True, False]
        mgr.engine.step.return_value = [ro]

        mgr.run_phase("inference", "PREFILL")

        split_vllm.torch.cuda.nvtx.range_pop.assert_called()


# =========================================================================== #
#  EngineManager.shutdown                                                      #
# =========================================================================== #

class TestEngineManagerShutdown:
    """Verify that shutdown releases Python object and flushes CUDA cache."""

    def test_shutdown_calls_empty_cache(self):
        with patch("split_vllm.EngineArgs"), \
             patch("split_vllm.LLMEngine") as mock_llm_cls:
            mock_llm_cls.from_engine_args.return_value = MagicMock()
            mgr = EngineManager(model="test-model")

        with patch("split_vllm.gc") as mock_gc:
            mgr.shutdown()

        mock_gc.collect.assert_called_once()
        split_vllm.torch.cuda.empty_cache.assert_called()

    def test_shutdown_deletes_engine_attribute(self):
        with patch("split_vllm.EngineArgs"), \
             patch("split_vllm.LLMEngine") as mock_llm_cls:
            mock_llm_cls.from_engine_args.return_value = MagicMock()
            mgr = EngineManager(model="test-model")

        with patch("split_vllm.gc"):
            mgr.shutdown()

        assert not hasattr(mgr, "engine")


# =========================================================================== #
#  ProfileReporter.report                                                      #
# =========================================================================== #

class TestProfileReporterReport:
    """Verify profiling output format and computed values."""

    def _run_report(self, prompt_tokens=100, gen_tokens=50):
        init = PhaseStats("INITIALIZATION", 2.1, "CPU+GPU")
        prefill = PhaseStats("PREFILL", 0.05, "GPU", 100, 1)
        decode = PhaseStats("DECODE", 0.9, "GPU", prompt_tokens, gen_tokens)
        buf = StringIO()
        with patch("sys.stdout", buf):
            ProfileReporter.report("out.txt", init, prefill, decode)
        return buf.getvalue()

    def test_output_contains_token_counts(self):
        out = self._run_report(prompt_tokens=100, gen_tokens=50)
        assert "100" in out
        assert "50" in out

    def test_output_contains_phase_labels(self):
        out = self._run_report()
        assert "INITIALIZATION" in out
        assert "PREFILL" in out
        assert "DECODE" in out

    def test_ratio_computed_correctly(self):
        out = self._run_report(prompt_tokens=100, gen_tokens=50)
        assert "0.500" in out

    def test_zero_prompt_tokens_avoids_division_by_zero(self):
        init = PhaseStats("INITIALIZATION", 1.0, "CPU+GPU")
        prefill = PhaseStats("PREFILL", 0.1, "GPU", 0, 0)
        decode = PhaseStats("DECODE", 0.5, "GPU", 0, 0)
        buf = StringIO()
        with patch("sys.stdout", buf):
            ProfileReporter.report("out.txt", init, prefill, decode)
        out = buf.getvalue()
        assert "0.000" in out

    def test_total_latency_is_prefill_plus_decode(self):
        init = PhaseStats("INITIALIZATION", 1.0, "CPU+GPU")
        prefill = PhaseStats("PREFILL", 0.25, "GPU", 10, 1)
        decode = PhaseStats("DECODE", 0.75, "GPU", 10, 20)
        buf = StringIO()
        with patch("sys.stdout", buf):
            ProfileReporter.report("out.txt", init, prefill, decode)
        out = buf.getvalue()
        assert "1.000" in out  # 0.25 + 0.75

    def test_output_path_printed(self):
        out = self._run_report()
        assert "out.txt" in out


# =========================================================================== #
#  InferencePipeline._load_input                                               #
# =========================================================================== #

class TestInferencePipelineLoadInput:
    """File reading, inline text, and empty-input paths."""

    def test_returns_empty_string_when_no_source(self):
        args = _make_args()
        result = InferencePipeline._load_input(args)
        assert result == ""

    def test_uses_inline_text(self):
        args = _make_args(text="inline content")
        result = InferencePipeline._load_input(args)
        assert "inline content" in result

    def test_reads_input_file(self, tmp_path):
        p = tmp_path / "input.txt"
        p.write_text("file content", encoding="utf-8")
        args = _make_args(input=str(p))
        result = InferencePipeline._load_input(args)
        assert "file content" in result

    def test_inline_text_aggressive_normalize(self):
        args = _make_args(
            text='["item1", "item2"]', aggressive_normalize=True
        )
        result = InferencePipeline._load_input(args)
        assert "[" not in result
        assert '"' not in result

    def test_inline_text_non_aggressive_preserves_structure(self):
        args = _make_args(
            text='{"key": "value"}', aggressive_normalize=False
        )
        result = InferencePipeline._load_input(args)
        assert '"key"' in result

    def test_text_takes_priority_over_input_file(self, tmp_path):
        """When both --text and --input are provided, --text wins."""
        p = tmp_path / "file.txt"
        p.write_text("from file", encoding="utf-8")
        args = _make_args(text="from text", input=str(p))
        result = InferencePipeline._load_input(args)
        assert "from text" in result

    def test_missing_input_file_raises(self):
        args = _make_args(input="/nonexistent/path/file.txt")
        with pytest.raises(FileNotFoundError):
            InferencePipeline._load_input(args)

    def test_aggressive_normalize_defaults_to_false_when_attr_absent(self):
        """Namespace without aggressive_normalize attr should not crash."""
        args = argparse.Namespace(text="hello", input=None)
        # aggressive_normalize attribute intentionally absent
        result = InferencePipeline._load_input(args)
        assert "hello" in result


# =========================================================================== #
#  InferencePipeline.run  (integration-level, all heavy deps mocked)          #
# =========================================================================== #

class TestInferencePipelineRun:
    """
    End-to-end pipeline run with all external dependencies mocked.
    Verifies that the orchestration wires components together correctly.
    """

    def _build_mocks(self, output_text="generated response"):
        """
        Return a tuple of (mock_engine_manager, mock_tokenizer) configured
        for a successful single-request run.
        """
        tokenizer = _make_tokenizer(
            encode_return=[1, 2, 3, 4, 5],
            decode_return=output_text,
        )
        prefill_stats = PhaseStats("PREFILL", 0.05, "GPU", 5, 1)
        decode_stats = PhaseStats("DECODE", 0.80, "GPU", 5, 10)

        mock_manager = MagicMock(name="EngineManager")
        mock_manager.run_phase.side_effect = [
            (prefill_stats, "partial"),          # PREFILL call
            (decode_stats, output_text),         # DECODE call
        ]
        return mock_manager, tokenizer

    def test_run_writes_output_file(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(
            prompt="Describe AI.", output=str(out_file)
        )
        mock_manager, tokenizer = self._build_mocks("AI is amazing.")

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        assert out_file.exists()
        assert out_file.read_text(encoding="utf-8") == "AI is amazing."

    def test_run_calls_add_request(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(
            prompt="Describe AI.", output=str(out_file)
        )
        mock_manager, tokenizer = self._build_mocks()

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        mock_manager.engine.add_request.assert_called_once()

    def test_run_calls_prefill_then_decode(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(
            prompt="Hello.", output=str(out_file)
        )
        mock_manager, tokenizer = self._build_mocks()

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        assert mock_manager.run_phase.call_count == 2
        calls = mock_manager.run_phase.call_args_list
        assert calls[0] == call("inference", "PREFILL")
        assert calls[1] == call("inference", "DECODE")

    def test_run_calls_shutdown(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(prompt="Hello.", output=str(out_file))
        mock_manager, tokenizer = self._build_mocks()

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        mock_manager.shutdown.assert_called_once()

    def test_run_calls_profile_report(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(prompt="Hello.", output=str(out_file))
        mock_manager, tokenizer = self._build_mocks()

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.ProfileReporter.report") as mock_report:
            InferencePipeline().run(args)

        mock_report.assert_called_once()

    def test_run_with_translation_mode(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(
            input=None,
            text="Hello world.",
            output=str(out_file),
            source="en",
            target="hi",
        )
        mock_manager, tokenizer = self._build_mocks("नमस्ते दुनिया।")

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        content = out_file.read_text(encoding="utf-8")
        assert "नमस्ते" in content

    def test_run_raises_value_error_on_missing_source_target(self, tmp_path):
        out_file = tmp_path / "result.txt"
        args = _make_args(output=str(out_file))  # no prompt, no source/target
        tokenizer = _make_tokenizer()

        with patch("split_vllm.EngineManager"), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer):
            with pytest.raises(ValueError, match="--source and --target"):
                InferencePipeline().run(args)

    def test_max_tokens_derived_from_prompt_length(self, tmp_path):
        """SamplingParams max_tokens must equal int(input_tokens * 1.3)."""
        out_file = tmp_path / "result.txt"
        args = _make_args(prompt="Hello.", output=str(out_file))
        mock_manager, tokenizer = self._build_mocks()
        # 10 prompt tokens → expected max_tokens = 13
        tokenizer.encode.return_value = list(range(10))

        captured_params = {}

        def capture_add_request(req_id, prompt, sampling_params):
            captured_params["sp"] = sampling_params

        mock_manager.engine.add_request.side_effect = capture_add_request

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.SamplingParams") as mock_sp_cls, \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        _, kwargs = mock_sp_cls.call_args
        assert kwargs["max_tokens"] == 13  # int(10 * 1.3)

    def test_sampling_params_temperature_zero(self, tmp_path):
        """Temperature must be 0 for deterministic inference."""
        out_file = tmp_path / "result.txt"
        args = _make_args(prompt="Hello.", output=str(out_file))
        mock_manager, tokenizer = self._build_mocks()

        with patch("split_vllm.EngineManager", return_value=mock_manager), \
             patch("split_vllm.get_tokenizer", return_value=tokenizer), \
             patch("split_vllm.SamplingParams") as mock_sp_cls, \
             patch("split_vllm.ProfileReporter.report"):
            InferencePipeline().run(args)

        _, kwargs = mock_sp_cls.call_args
        assert kwargs["temperature"] == 0.0


# =========================================================================== #
#  build_arg_parser                                                            #
# =========================================================================== #

class TestBuildArgParser:
    """Verify that the CLI parser accepts valid inputs and rejects bad ones."""

    def _parse(self, argv):
        parser = build_arg_parser()
        return parser.parse_args(argv)

    def test_output_is_required(self):
        parser = build_arg_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_minimal_valid_args(self):
        args = self._parse(["--output", "out.txt"])
        assert args.output == "out.txt"
        assert args.model == "Qwen/Qwen2.5-1.5B"  # default

    def test_custom_model(self):
        args = self._parse([
            "--output", "out.txt", "--model", "Qwen/Qwen2.5-7B"
        ])
        assert args.model == "Qwen/Qwen2.5-7B"

    def test_prompt_argument(self):
        args = self._parse([
            "--output", "out.txt", "--prompt", "Describe AI."
        ])
        assert args.prompt == "Describe AI."

    def test_source_and_target(self):
        args = self._parse([
            "--output", "out.txt", "--source", "en", "--target", "hi"
        ])
        assert args.source == "en"
        assert args.target == "hi"

    def test_input_file_argument(self):
        args = self._parse([
            "--output", "out.txt", "--input", "doc.txt"
        ])
        assert args.input == "doc.txt"

    def test_text_argument(self):
        args = self._parse([
            "--output", "out.txt", "--text", "some text"
        ])
        assert args.text == "some text"

    def test_aggressive_normalize_flag_default_false(self):
        args = self._parse(["--output", "out.txt"])
        assert args.aggressive_normalize is False

    def test_aggressive_normalize_flag_sets_true(self):
        args = self._parse([
            "--output", "out.txt", "--aggressive-normalize"
        ])
        assert args.aggressive_normalize is True

    def test_defaults_are_none_for_optional_args(self):
        args = self._parse(["--output", "out.txt"])
        assert args.input is None
        assert args.text is None
        assert args.prompt is None
        assert args.source is None
        assert args.target is None