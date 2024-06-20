from .loader import load_data, Bin, Flower
import numpy as np
from .evaluate_fitness import evaluate
import time
    
class Block:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.matrix = np.zeros((width, height))
        self.current_x = 0
        self.debug_elements = []
    
    def __repr__(self):
        return f"{self.debug_elements}"
        
    def can_place(self, flower: Flower) -> bool:
        return self.current_x + flower.width <= self.width

    def place(self, flower: Flower, flower_number) -> bool:
        if self.can_place(flower):
            self.matrix[self.current_x:self.current_x+flower.width, 0:flower.height] = np.ones([flower.width, flower.height])
            self.debug_elements.append((str(flower), str(flower_number)))
            self.current_x +=  flower.width
            return True
        return False

class HFFBin(Bin):
    def __init__(self, bin: Bin):
        self.width = bin.width
        self.height = bin.height
        self.current_y = 0
        self.matrix = np.zeros((bin.width, bin.height))
        self.debug_elements = []

    def can_place(self, block: Block) -> bool:
        return self.current_y + block.height <= self.height

    def place(self, block:Block, block_number) -> bool:
        if self.can_place(block):
            self.matrix[0:block.width, self.current_y:self.current_y+block.height] = block.matrix
            self.debug_elements.append((block, block_number))
            self.current_y +=  block.height
            return True
        return False
    
    def get_packing_density(self):
        return np.count_nonzero(self.matrix==1) / (self.height*self.width)
        
    def is_not_used(self):
        return np.all(self.matrix == 0)


def group_flowers(flowers: list[Flower], width):
    blocks = []
    for i, flower in enumerate(flowers):
        if len(blocks) == 0:
            blocks.append(Block(width, flower.height))
        placed = False
        for block in blocks:
            if block.place(flower, i):
                placed = True
                break
        if not placed:
            blocks.append(Block(width, flower.height))
            blocks[-1].place(flower, i)
    return blocks


def algorithm_HFF(flowers:list[Flower], bins:list[Bin]):
    flowers.sort(key=lambda flower: flower.height, reverse=True)
    bins = [HFFBin(bin) for bin in bins]

    grouped_flowers = group_flowers(flowers, bins[0].width)
    for i, block in enumerate(grouped_flowers):
        for bin in bins:
            if bin.place(block, i):
                break

    return bins

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
    
    return bins

def run_algortihm_HFF_HBF(bins_filename_, flowers_filename_):
    
    loaded_bins, loaded_flowers = load_data(bins_filename=bins_filename_, flowers_filename=flowers_filename_)
    start_time = time.time()
    HBF_results = algorithm_HBF(loaded_flowers, loaded_bins)
    end_time = time.time()
    HBF_execution_time = end_time - start_time
    HBF_f = evaluate([bin.matrix for bin in HBF_results], False)
    loaded_bins, loaded_flowers = load_data(bins_filename=bins_filename_, flowers_filename=flowers_filename_)
    start_time = time.time()
    HFF_results = algorithm_HFF(loaded_flowers, loaded_bins)
    end_time = time.time()
    HFF_execution_time = end_time - start_time
    HFF_f = evaluate([bin.matrix for bin in HFF_results], False)
    print(f"HBF: fitenss: {HBF_f[0]}, bins: {HBF_f[1]}, czas: {HBF_execution_time}")
    print(f"HFF: fitenss: {HFF_f[0]}, bins: {HFF_f[1]}, czas: {HFF_execution_time}")

if __name__ == "__main__":
    print("=========")
    run_algortihm_HFF_HBF('size10/size10bins.csv', 'size10/count100flowers/count100flowers1.csv')
    print("=========")
    run_algortihm_HFF_HBF('size10/size10bins.csv', 'size10/count100flowers/count100flowers2.csv')
    print("=========")
    run_algortihm_HFF_HBF('size10/size10bins.csv', 'size10/count100flowers/count100flowers3.csv')