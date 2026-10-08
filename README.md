*This project has been created as part of the 42 curriculum by srosu.*

# Call Me Maybe

## Description

The purpose of this project is to translate a natural language prompt into a
structured function call. Given the prompt `What is the sum of 2 and 3?`, the
program does not answer `5`. It answers which function to call and with which
arguments:

```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {"a": 2.0, "b": 3.0}
}
```

The problem is that the model used in this project, `Qwen/Qwen3-0.6B`, is
small. It has around **600 million** parameters, a vocabulary of **151 936**
tokens and a context window of **32 768** tokens. That makes it light enough
to run on a local machine, without a GPU and without an internet connection
once it is downloaded. The price is reliability: when it is simply asked to
write JSON, a model of this size often produces invalid output (a missing
quote, a wrong type, an invented key).

The solution used here is called **constrained decoding**. Instead of
letting the model write freely and hoping for valid JSON, the program checks
every candidate token and forbids the ones that would break the expected
structure. The model still chooses the function and the values, but it can
only write something valid. This method is described in the
[Algorithm explanation](#algorithm-explanation) section.

## Instructions

Python 3.14 and all the dependencies are installed by `uv`.

| Command | Description |
| - | - |
| `make install` | installs the dependencies with `uv sync` |
| `make run` | runs the project with `uv run python3 -m src` |
| `make debug` | runs the project in Python's debugger with `uv run python3 -m pdb -m src` |
| `make lint` | runs `flake8 .` and `mypy .` with the flags required by the subject |
| `make lint-strict` | runs `flake8 .` and `mypy . --strict` |
| `make clean` | removes `__pycache__` and `.mypy_cache` |

By default, the program reads `data/input/function_calling_tests.json` and
`data/input/functions_definition.json`, and writes
`data/output/function_calls.json`. These paths can be changed:

| Argument | Value |
| - | - |
| `--input` | `<prompts_file>` |
| `--functions_definition` | `<functions_definition_file>` |
| `--output` | `<output_file>` |

```bash
uv run python -m src --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calls.json
```

`--help` (or `-h`) displays the available options.

### Running on a 42 machine

The dependencies and the model need several gigabytes, more than the home
quota. These variables move everything to `sgoinfre`:

```bash
export UV_CACHE_DIR="/sgoinfre/students/<login>/uv-cache"
export UV_PROJECT_ENVIRONMENT="/sgoinfre/students/<login>/callmemaybe-venv"
export UV_PYTHON_INSTALL_DIR="/sgoinfre/students/<login>/uv-python"
export HF_HOME="/sgoinfre/students/<login>/huggingface"
```

## Algorithm explanation

Each prompt goes through two steps: **function selection**, then
**parameter extraction**. The `run` method of `pipeline.py` drives them. For
each prompt it encodes the text, selects a function, extracts its arguments,
validates the result with Pydantic, displays the progress, and finally
returns a list of `FunctionCall` objects that are written to the JSON file.

Both steps rely on the `generate` method of `decoder.py`, where constrained
decoding happens. At each iteration:

1. The model receives the token ids and returns one score (logit) for every
   token of the vocabulary.
2. An `allowed` array is created, filled with `-inf`.
3. For each token, the program checks whether the text generated so far plus
   this token **can still become** a valid value. If so, the logit of the
   token is copied into `allowed`.
4. `np.argmax` picks the best allowed token. A forbidden token keeps `-inf`
   and can never win.
5. The token is appended to the prompt and the loop starts again.

Each constraint is made of two functions. `is_valid` tells whether a text can
still become correct, and `is_done` tells whether it is finished. They are
different: for a number, `26` is valid but not finished, because it could
still become `265`. The value is finished only when its closing character
(`,` or `}`) is written.

The fixed parts of the JSON (braces, keys, quotes, colons) are written by
the code. The model only fills in the function name and the values.

Each parameter type has its own rule:

- **string**: any text up to the closing quote, with valid JSON escapes only
  (`\"`, `\\`, `\n`...). Raw control characters are rejected.
- **number**: an optional minus sign, digits, and an optional decimal part.
- **integer**: an optional minus sign and digits.
- **boolean**: only a prefix of `true` or `false`.

For the function name, a text is valid if it is the beginning of an existing
function name, so the model can never invent a function.

A `complete` function avoids useless calls to the model when only one outcome
is left. If a boolean starts with `t`, the result can only be `true`. If only
one function name starts with `fn_g`, the rest of the name is written
directly.

The loop stops when the value is finished, or raises an error when no token
is valid or when the token limit is reached. The limit is 50 tokens, except
for strings, where it is the length of the prompt plus 20. This prevents
infinite loops if the model never writes the expected end.

## Design decisions

- **One shared `Decoder`.** It holds the model, the decoded vocabulary, an
  `encode` helper and the `generate` method. The function selector and the
  parameter extractor both use the same instance, so the vocabulary is
  decoded only once, at startup.
- **Pydantic everywhere.** Every class is a `BaseModel`. The input files are
  validated when they are loaded. `create_model` builds, at run time, one
  validation model per function, because the functions are unknown before the
  definition file is read. Each generated call is checked against it in
  strict mode: no missing argument, no extra key, no wrong type, no infinite
  number.
- **A failed prompt is skipped**, and the reason is printed on stderr. The
  output file therefore only contains valid calls.
- **Exit codes**: `0` when the output file is written, `1` on error, `130`
  when the user interrupts the program.
- **Progress goes to stderr**, the final summary to stdout and the result to
  the output file. A closed pipe (`make run | head`) cannot stop the
  generation.
- **An input file is never overwritten.** If `--output` points to one of the
  input files, the program stops before loading the model.
- **Strict definition file.** An unknown parameter type, an invalid function
  or parameter name, or a duplicated function name makes the whole file
  invalid. The program raises a custom `CallMeError` and displays where the
  error is. Extra keys such as `returns` are ignored.
- **An empty prompt is rejected** when the file is loaded.
- **A function is always selected**, even when the prompt is vague or
  unrelated to any function. The program has no "no function" answer.

## Performance analysis

**Accuracy.** On the 11 provided prompts, 10 are fully correct. The remaining
one is `Replace all vowels in 'Programming is fun' with asterisks`: the
function and the regex are right, but the model writes `*****` instead of `*`
as the replacement. Constrained decoding guarantees the form of the output
(valid JSON, right types), not its meaning.

**Speed.** On my personal machine (MacBook, Apple M4, 16 GB of RAM), the 11
prompts take about 32 seconds. This depends on the machine and on the number
of tokens to process.
On a 42 machine (CPU Only | Intel(R) Core(TM) i5-7500 CPU @ 3.40GHz)
, the same 11 prompts take about 3 minutes 33 seconds, under the 5-minute limit of the subject.

The SDK keeps no cache between two calls, so each generated token makes the
model process the whole prompt again. The cost of a prompt is roughly its
length multiplied by the number of generated tokens. Two optimizations follow
from this:

- **Fewer examples.** The extraction prompt first contained 7 examples. Four
  of them were redundant, and removing them did not change any output on a
  set of 20 reference prompts.
- **The `complete` function**, which skips the model when the end of a value
  is already determined.

On a set of 5 prompts, these two changes reduced the time from 36 to 21
seconds.

More examples would probably improve accuracy on some prompts, but every
example makes every generated token slower. The current prompt is a
compromise between the two.

## Challenges faced

- **Raw vocabulary versus decoded text.** In the vocabulary file, a space is
  stored as `Ġ` and a line break as `Ċ`. My first rules compared this raw
  text with real text, so they accepted tokens they should have refused. The
  vocabulary is now decoded once with the SDK, and the rules only see real
  text.
- **Token healing.** When the code forces a piece of text, the model can no
  longer use the tokens that would have merged this piece with what follows.
  This caused three bugs:
  - a space in front of some values (`" *"`), because the opening quote was
    forced and the model fell back on a token starting with a space;
  - a function whose name is the beginning of another one (`fn_add` and
    `fn_add_numbers`) could never be selected, because the model wanted to
    close the name with the token `",` while only `"` was allowed;
  - the minus sign of negative numbers was lost, because a space was forced
    after the colon while the model writes ` -` as a single token.
- **JSON escapes in regular expressions.** The model writes `\\d+` in JSON.
  The value has to be decoded with `json.loads` to get the real regex `\d+`,
  and the string rule has to know that `\"` does not close the string.
- **Known limitations**, which come from the model more than from the rules:
  - an empty string is returned as a single space;
  - a double quote inside a value closes it too early;
  - characters encoded over several tokens (some emojis) cannot be
    reproduced: a partial character decodes to U+FFFD, these tokens are
    rejected, and the model writes another character instead;
  - when an argument is missing from the prompt, the model invents one.

## Testing strategy

- **Edge cases**, grouped in five families: command line (unknown argument,
  missing dependency, Ctrl+C), file reading (missing file, invalid JSON,
  wrong encoding, no permission), file structure (wrong root type, missing
  key, unknown type, duplicated or invalid names), generation (empty string,
  special characters, negative and large numbers, booleans, mixed types,
  ambiguous prompts, prompt injection) and output (missing folders,
  unwritable path, output equal to an input).
- **A reference set of 20 prompts**, run again after each change to the
  prompt or to the rules, to check that no output regressed.
- **A clean clone** followed by `uv sync` and a full run, to check that the
  repository is self-sufficient.
- **Static checks**: `flake8` and `mypy --strict` both pass.

## Example usage

```bash
$ make run
[1/11] What is the sum of 2 and 3? -> fn_add_numbers
{'a': 2.0, 'b': 3.0}

[2/11] What is the sum of 265 and 345? -> fn_add_numbers
{'a': 265.0, 'b': 345.0}

[3/11] Greet shrek -> fn_greet
{'name': 'shrek'}
...
Total time: 0 minutes 32 seconds
11/11 prompts processed
0 skipped
Output: data/output/function_calls.json
```

Extract of `data/output/function_calls.json`:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Replace all numbers in \"Hello 34 I'm 233 years old\" with NUMBERS",
    "name": "fn_substitute_string_with_regex",
    "parameters": {
      "source_string": "Hello 34 I'm 233 years old",
      "regex": "\\d+",
      "replacement": "NUMBERS"
    }
  }
]
```

An invalid input gives a clear message and exit code 1:

```bash
$ uv run python -m src --input missing.json
Error: file not found: missing.json
```

## Resources

- [Qwen3-0.6B model card](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702),
  the paper behind constrained decoding libraries
- [Pydantic documentation](https://docs.pydantic.dev/)
- [NumPy documentation](https://numpy.org/doc/)
- [uv documentation](https://docs.astral.sh/uv/)
- [RFC 8259, the JSON format](https://www.rfc-editor.org/rfc/rfc8259)
- [PEP 257, docstring conventions](https://peps.python.org/pep-0257/)

### Use of AI

I used an AI assistant (Claude) during this project for:

- explanations of the concepts: tokenization, logits, constrained decoding,
  Pydantic features;
- suggest a file structure;
- an edge-case test campaign on the finished program, which produced the
  list of the possible problems that I then fixed myself
- proofreading and completing this README.

The code was entirely written and debugged by me.