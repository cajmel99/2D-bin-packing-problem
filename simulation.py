from arguments import create_parser
from algorithms import run_algortihm_HFF_HBF, run_genetic_algorithm, solve_sa, generate_and_save_data, random_algorith2, evaluate

if __name__ == '__main__':
    args = None
    try:
        args = create_parser()
    except Exception as e:
        print(e)
        exit(1)

    match args.algorith:
        case "GENERATE_DATA":
            generate_and_save_data(
                args.bins_filename,
                args.flowers_filename,
                args.data_folder,
                args.num_bins,
                args.bin_max_width,
                args.bin_max_height,
                args.num_flowers,
                args.flower_min_width,
                args.flower_max_width,
                args.flower_min_height,
                args.flower_max_height
            )
        case "SA":
            print(f'\nYou\'ve chosen SA as the algorithm!')
            print(f'\nLoaded files:\n|-> Bins file: {args.source_bins} \n|-> Flowers file: {args.source_flowers}')
            print(f'\nParams:\n|-> Max iterations: {args.iterations} \n|-> Initial temperature: {args.temperature} \n|-> Cooling rate: {args.cooling}')
            sim_result = solve_sa(args.source_bins, args.source_flowers, int(args.iterations), int(args.temperature), float(args.cooling))
        case "GREEDY":
            run_algortihm_HFF_HBF(args.source_bins, args.source_flowers)
        case "RANDOM":
            results = random_algorith2(args.source_bins, args.source_flowers)
            print(results)
            print(f"Number of used bins: ", len(results))
            final_fitness = evaluate(results, True)
            print("Final fitness: ", final_fitness)
        case "GA":
            run_genetic_algorithm(
                int(args.pop_size),
                int(args.generations),
                int(args.tournament),
                float(args.cross),
                float(args.mut),
                args.source_flowers,
                args.source_bins,
                args.results_path,
                int(args.iterations)
            )
        case _:
            print("ASDASD")



