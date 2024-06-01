from loader import load_data, print_data, Bin, Flower
# from load_data import bins_matrices, flowers_pools
import numpy as np
    

class HBFBin(Bin):
    def __init__(self, bin: Bin):
        self.width = bin.width
        self.height = bin.height
        self.current_y = 0
        self.first_flower_in_row_height = 0
        self.current_x = 0
        self.matrix=np.zeros((bin.width, bin.height))

    def can_place(self, flower):
        return self.current_x + flower.width <= self.width

    def place(self, flower: Flower):
        if self.can_place(flower):
            if self.current_x == 0:
                self.first_flower_in_row_height = flower.height
            self.matrix[self.current_x:self.current_x + flower.width, self.current_y:self.current_y + flower.height] = np.ones([flower.width, flower.height])
            self.current_x += flower.width
            return True
        return False

    def can_open_new_block(self, flower):
        return self.current_y + flower.height + self.first_flower_in_row_height <= self.height

    def open_new_block(self, flower):
        if self.can_open_new_block(flower):
            self.current_y += self.first_flower_in_row_height
            self.current_x = 0
            self.place(flower)
            return True
        return False

    def can_put_in_current_bin(self, flower: Flower):
        return self.current_y + flower.height <= self.height

    def __repr__(self):
        return f'width={self.width},height={self.height},current_y={self.current_y})'
    
    def get_packing_density(self):
        return np.count_nonzero(self.matrix==1) / (self.height*self.width)
    
    def is_not_used(self):
        return np.all(self.matrix == 0)


def algorithm_HBF(flowers, bins):
    flowers.sort(key=lambda flower: flower.height, reverse=True)
    bins = [HBFBin(bin) for bin in bins]
    current_bin_index = 0

    for i, flower in enumerate(flowers):
        placed = False
        if bins[current_bin_index].can_put_in_current_bin(flower):
            if bins[current_bin_index].place(flower):
                placed = True
        if not placed:
            if not bins[current_bin_index].open_new_block(flower):
                current_bin_index+=1
                bins[current_bin_index].open_new_block(flower)

    # u_bins = [b for b in bins if not b.is_not_used()]
    # used_bins = len(u_bins)
    # print(used_bins)
    # avg_packing_density = sum(b.get_packing_density() for b in u_bins) / used_bins
    # xx = 1 / used_bins + avg_packing_density
    # print(f"========{xx}=========")

    # for i, bin in enumerate(bins):
    #     filename = f"bin_{i+1}.csv"
    #     np.savetxt(filename, np.rot90(bin.matrix), delimiter=",", fmt="%d")
    
    return bins


if __name__ == "__main__":
    loaded_bins, loaded_flowers = load_data(bins_filename='bins2.csv', flowers_filename='flowers2.csv')
    results = algorithm_HBF(loaded_flowers, loaded_bins)

    from evaluate_fitness import evaluate

    x = evaluate([bin.matrix for bin in results], False)
    print(x)
