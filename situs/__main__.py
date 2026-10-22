"""Executable module entrypoint for python -m situs."""

import sys

def main():
    try:
        from situs.cli import main as cli_main
        cli_main()
    except (ImportError, ModuleNotFoundError) as e:
        sys.stderr.write(f"\n[Situs] Missing required Python dependency: {e}\n\n")
        sys.stderr.write("To install all dependencies for this device, run in PowerShell:\n")
        sys.stderr.write("    pip install -e .\n")
        sys.stderr.write("or:\n")
        sys.stderr.write("    pip install -r requirements.txt\n\n")
        sys.exit(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
