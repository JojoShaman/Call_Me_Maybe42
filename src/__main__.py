"""Entry point: turn natural language prompts into function calls."""
import sys
try:
    from llm_sdk import Small_LLM_Model
    from .data import Data
    from .errors import CallMeError
    from .cli import arguments
    from .io_utils import write_output
    from time import perf_counter
    from .decoder import Decoder
    from .extractors import ExtractParameter, SelectFunction
    from .pipeline import Pipeline
except KeyboardInterrupt:
    print("Program interrupted by user.", file=sys.stderr)
    sys.exit(130)
except ModuleNotFoundError as e:
    missing = getattr(e, "name", None) or str(e)
    print(f'Error: missing dependency "{missing}". '
          "Run 'uv sync' or 'make install'.", file=sys.stderr)
    sys.exit(1)
except ImportError as e:
    print(f"Error: import failed: {e}. "
          "Run the program with 'uv run python -m src'.", file=sys.stderr)
    sys.exit(1)

def main() -> int:
    """Run the whole program and return its exit code.

    Parses the command line, loads the input files and the model,
    processes every prompt and writes the output file.

    Returns:
        0 when the output file was written, 1 when an error prevented
        it, 130 when the user interrupted the program.
    """
    try:
        args = arguments()
        out = args.output.resolve()
        if out in (args.input.resolve(),
                   args.functions_definition.resolve()):
            raise CallMeError(
                f"--output would overwrite an input file: {args.output}")
        d = Data.open_files(args.input, args.functions_definition)
        start_time = perf_counter()
        try:
            m = Small_LLM_Model()
        except Exception as e:
            raise CallMeError(f"cannot load the model: {e}")
        decoder = Decoder(model=m)
        pipeline = Pipeline(
            data=d, deco=decoder,
            function=SelectFunction(names=list(d.function), decoder=decoder),
            param=ExtractParameter(decoder=decoder))
        result = pipeline.run()
        end_time = perf_counter()
        write_output(args.output, result)
    except CallMeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Program interrupted by user.', file=sys.stderr)
        return 130
    except Exception as e:
        print(f'Unexpected error occurred: {e}', file=sys.stderr)
        return 1
    minutes, seconds = divmod(round(end_time - start_time), 60)
    total = len(d.prompts)
    print(f'Total time: {minutes} minutes {seconds} seconds')
    print(f'{len(result)}/{total} prompts processed\n'
          f'{total - len(result)} skipped\n'
          f'Output: {args.output}')

    return 0


if __name__ == '__main__':
    sys.exit(main())
