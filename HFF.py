from loader import load_data, print_data, Bin, Flower
# from load_data import bins_matrices, flowers_pools
import numpy as np
import matplotlib.pyplot as plt
    
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
    loaded_bins, loaded_flowers = load_data(bins_filename='count100bins/size10/size10bins.csv', flowers_filename='count100bins/size10/count50flowers.csv')
        
    heights = [f.height for f in loaded_flowers]
    areas = [f.height*f.width for f in loaded_flowers]
    results = algorithm_HFF(loaded_flowers, loaded_bins)

    # Tworzenie histogramów z większą liczbą przedziałów (np. 20)
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Histogram dla pól powierzchni
    axs[0].hist(areas, bins=20, color='skyblue', edgecolor='black')
    axs[0].set_title('Histogram pól powierzchni elementów')
    axs[0].set_xlabel('Pole powierzchni')
    axs[0].set_ylabel('Częstotliwość')

    # Histogram dla wysokości
    axs[1].hist(heights, bins=20, color='lightgreen', edgecolor='black')
    axs[1].set_title('Histogram wysokości elementów')
    axs[1].set_xlabel('Wysokość')
    axs[1].set_ylabel('Częstotliwość')

    plt.tight_layout()
    plt.show()


    from evaluate_fitness import evaluate

    x = evaluate([bin.matrix for bin in results], True)
    print(x)
