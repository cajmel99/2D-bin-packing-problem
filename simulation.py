import sys, getopt
from simulated_annealing import solve_sa

def main(argv):
    if len(argv) < 3:
        print(f'Too little arguments!')
        print(f'\nProper use:\n\t$ python simulation.py [algorithm] [bins_file] [flowers_file]')
        print(f'\nWhere [algorithm] is one of: RA, HBF, HFF, GA, SA\n')

        return

    if argv[0] == 'SA':
        if len(argv) < 6:
            print(f'Too little arguments for simulated annealing algorithm!')
            print(f'\nProper use:\n\t$ python simulation.py SA [bins_file] [flowers_file] [max_iterations] [initial_temperature] [cooling_rate]')

            return

        print(f'\nYou\'ve chosen SA as the algorithm!')
        print(f'\nLoaded files:\n|-> Bins file: {argv[1]} \n|-> Flowers file: {argv[2]}')
        print(f'\nParams:\n|-> Max iterations: {argv[3]} \n|-> Initial temperature: {argv[4]} \n|-> Cooling rate: {argv[5]}')
        sim_result = solve_sa(argv[1], argv[2], int(argv[3]), int(argv[4]), float(argv[5]))


if __name__ == "__main__":
   main(sys.argv[1:])
