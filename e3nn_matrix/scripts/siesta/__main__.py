import argparse
from e3nn_matrix.scripts.siesta.analyze_MD import visualize_performance_table

parser = argparse.ArgumentParser(
    prog = 'Siesta utils',
    description = """
    Should contain a set of siesta utils to set up MD runs using a model, evaluate performance, etc.
    For now it only contains a script to analyze the output of a MD run.
    """,
    epilog = 'Text at the bottom of help'
)

parser.add_argument('out_files', nargs='+', help='Paths to the output files of the MD runs.')
parser.add_argument('--precision', '-p', default=3, type=int, help='Number of decimal places to show in the table.')
parser.add_argument('--save_path', '-s', default=None, help='Path to save the HTML table to. If not provided, the table is displayed in the browser.')

args = parser.parse_args()

visualize_performance_table(args.out_files, precision=args.precision, save_path=args.save_path, )