import argparse

class MissingArgumentException(Exception):
    def __init__(self, message="One or more required arguments are missing."):
        self.message = message
        super().__init__(self.message)

class ConfigFileException(Exception):
    def __init__(self, message="One or more required arguments are missing."):
        self.message = message
        super().__init__(self.message)

def config_file_read(file_path):
    dictionary = {}
    with open(file_path, "r") as file:
        for line in file:
            key, value = line.strip().split("=")
            dictionary[key] = value
    return dictionary

def sa_command(args):
    if args.config:
        configs = config_file_read(args.config)
        for key in configs:
            try:
                getattr(args, key)
            except:
                raise ConfigFileException()
            setattr(args, key, configs[key])
        if not all([args.source_bins, args.source_flowers, args.temperature, args.iterations, args.cooling]):
            raise ConfigFileException()
    elif all([args.source_bins, args.source_flowers, args.temperature, args.iterations, args.cooling]):
        pass
    else:
        raise MissingArgumentException("SA_ERROR")

def ga_command(args):
    if args.config:
        configs = config_file_read(args.config)
        for key in configs:
            try:
                getattr(args, key)
            except:
                raise ConfigFileException()
            setattr(args, key, configs[key])
        if not all([
            args.source_bins,
            args.source_flowers,
            args.pop_size,
            args.generations,
            args.tournament,
            args.cross,
            args.mut,
            args.results_path,
            args.iterations
        ]):
            raise ConfigFileException()
    elif all([
        args.source_bins,
        args.source_flowers,
        args.pop_size,
        args.generations,
        args.tournament,
        args.cross,
        args.mut,
        args.results_path,
        args.iterations
    ]):
        pass
    else:
        raise MissingArgumentException("GA_ERROR")

def greedy_command(args):
    print(f"GREEDY: source_bins={args.source_bins}, source_flowers={args.source_flowers}")

def generate_data(args):
    print(f"Generate data: bins={args.bins_filename}, flowers={args.flowers_filename}")
def create_parser():
    parser = argparse.ArgumentParser(description='Main program', formatter_class=argparse.RawTextHelpFormatter)
    subparsers = parser.add_subparsers(dest='algorith', help='Available commands')

    # Command 1 parser
    parser_command1 = subparsers.add_parser('SA', help='Command 1 help', formatter_class=argparse.RawTextHelpFormatter)
    parser_command1.add_argument(
        '--config',
        type=str,
        help='''Ścieżka do pliku konfiguracyjnego, format pliku konfigóracyjnego to:
source_bins=size10/size10bins.csv
source_flowers=size10/count50flowers/count50flowers1.csv
iterations=50
temperature=999
cooling=5'''
    )
    parser_command1.add_argument('--source_bins', type=str, help='Path to the first source file')
    parser_command1.add_argument('--source_flowers', type=str, help='Path to the second source file')
    parser_command1.add_argument('--iterations', type=int, help='Number of iterations')
    parser_command1.add_argument('--temperature', type=float, help='Initial temperature')
    parser_command1.add_argument('--cooling', type=float, help='Initial temperature')
    parser_command1.set_defaults(func=sa_command)

    # Command 2 parser
    parser_command2 = subparsers.add_parser('GREEDY', help='Command 2 help')
    parser_command2.add_argument('--source_bins', type=str, help='Path to the first source file', required=True)
    parser_command2.add_argument('--source_flowers', type=str, help='Path to the second source file', required=True)
    parser_command2.set_defaults(func=greedy_command)

    # Command 3 parser
    parser_command3 = subparsers.add_parser('GA', help='Command 2 help')
    parser_command3.add_argument(
        '--config',
        type=str,
        help='''Ścieżka do pliku konfiguracyjnego, format pliku konfigóracyjnego to:
    source_bins=size10/size10bins.csv
    source_flowers=size10/count50flowers/count50flowers1.csv
    iterations=50
    temperature=999
    cooling=5'''
    )
    parser_command3.add_argument('--source_bins', type=str, help='Path to the first source file')
    parser_command3.add_argument('--source_flowers', type=str, help='Path to the second source file')
    parser_command3.add_argument('--pop_size', type=int, help='Path to the first source file')
    parser_command3.add_argument('--generations', type=int, help='Path to the second source file')
    parser_command3.add_argument('--tournament', type=int, help='Path to the first source file')
    parser_command3.add_argument('--cross', type=float, help='Path to the second source file')
    parser_command3.add_argument('--mut', type=float, help='Path to the second source file')
    parser_command3.add_argument('--results_path', type=str, help='Path to the second source file')
    parser_command3.add_argument('--iterations', type=int, help='Path to the second source file')
    parser_command3.set_defaults(func=ga_command)

    # Command 4 parser
    parser_command4 = subparsers.add_parser('GENERATE_DATA', help='GENERATE_DATA')
    parser_command4.add_argument('--bins_filename', type=str, help='Path to the first source file', required=True)
    parser_command4.add_argument('--flowers_filename', type=str, help='Path to the second source file', required=True)
    parser_command4.add_argument('--data_folder', type=str, help='Path to the second source file', required=True)
    parser_command4.add_argument('--num_bins', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--bin_max_width', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--bin_max_height', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--num_flowers', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--flower_min_width', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--flower_max_width', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--flower_min_height', type=int, help='Path to the second source file', required=True)
    parser_command4.add_argument('--flower_max_height', type=int, help='Path to the second source file', required=True)
    parser_command4.set_defaults(func=generate_data)

    args = parser.parse_args()
    try:
        args.func(args)
    except MissingArgumentException as e:
        if e.message=="SA_ERROR":
            raise MissingArgumentException(parser_command1.print_help())
        elif e.message=="GA_ERROR":
            raise MissingArgumentException(parser_command3.print_help())
        else:
            raise MissingArgumentException("FDSFSD")
    except MissingArgumentException as e:
        raise ConfigFileException

    return args