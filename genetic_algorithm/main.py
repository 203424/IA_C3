from random import randint, uniform, random, sample
from itertools import combinations, product

class genetic_algorithm:
    def __init__(self, num_layers, num_neurons,pop_size,num_generations,mut_rate, mut_rate_pos, mut_rate_layer, mut_rate_neurons,mut_rate_f_activation, cross_rate):
        self.num_layers = num_layers #rango ej. (3,6)
        self.num_neurons = num_neurons #rango ej. (64,256)
        self.function = ['relu','softmax','sigmoide']
        self.pop_initial = randint(2,pop_size)
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mut_rate
        self.mutation_pos = mut_rate_pos
        self.mutation_layer = mut_rate_layer
        self.mutation_n_neurons = mut_rate_neurons
        self.mutation_f_activation = mut_rate_f_activation
        self.crossover_rate = cross_rate
        self.population = []
    
    def calculate_aptitude(self, individual): #deberá cambiarse por la funcion a utilizar
        total_neurons = 0
        for gen in individual:
            total_neurons += gen[0]
        return total_neurons

    def code_individual(self):
        individual = []
        for _ in range(round(uniform(*self.num_layers))):
            individual.append([round(uniform(*self.num_neurons)),self.function[round(uniform(0,len(self.function)-1))]])
        return individual

    def generate_population(self):
        for _ in range(self.pop_initial):
            individual = self.code_individual()
            if individual not in self.population:
                self.population.append(individual)
    
    def select_parents(self):
        pop_sorted = sorted(list(map(lambda x,y:[x,y], self.fitness,self.population)), reverse=True)
        mas_aptos = list(map(lambda x:x[1],pop_sorted[:len(pop_sorted)//2]))
        menos_aptos = list(map(lambda x:x[1],pop_sorted[len(pop_sorted)//2:]))

        parents_0 = list(combinations(mas_aptos, 2))
        parents_1 = list(product(mas_aptos, menos_aptos))

        parents_0.extend(parents_1)

        return parents_0

    def crossover(self):
        self.children = []
        self.fitness = [self.calculate_aptitude(x) for x in self.population]
        parents = self.select_parents()
        for p in parents:
            g_parent = [[], []]
            child_1, child_2 = [], []
            crosspoint_max = min(len(p[0]), len(p[1])) - 1
            crosspoints = []
            for i in range(2):
                crosspoint_max = randint(1,crosspoint_max)
                crosspoints_i = sorted(sample(range(1, len(p[i])), crosspoint_max))
                crosspoints.append(crosspoints_i)
            crosspoints = [crosspoints[0], crosspoints[1] + [len(p[1])]]
            for i in range(2):
                parent_frags = []
                inicio = 0
                for j in range(crosspoint_max):
                    frag = p[i][inicio:crosspoints[i][j]]
                    parent_frags.append(frag)
                    inicio = crosspoints[i][j]
                parent_frags.append(p[i][inicio:])
                g_parent[i] = parent_frags
            for i in range(len(g_parent[0])):
                if i % 2 == 0:
                    child_1, child_2 = child_1 + g_parent[0][i], child_2 + g_parent[1][i]
                else:
                    child_1, child_2 = child_1 + g_parent[1][i], child_2 + g_parent[0][i]
            self.children += [child_1, child_2]

    def mutate(self):
        for child in self.children:
            if random() < self.mutation_rate:
                child = self.mutate_child(child)
            self.population.append(child)

    def mutate_child(self, child):
        mutate_child = child.copy()
        for pos_gen in range(len(mutate_child)):
            if random() < self.mutation_pos:
                mutate_child = self.mutate_position(mutate_child, pos_gen)
            if random() < self.mutation_layer:
                mutate_child = self.mutate_layer(mutate_child, pos_gen)
        return mutate_child

    def mutate_position(self, child, pos_gen):
        random_pos = round(uniform(0, len(child) - 1))
        gen = child[pos_gen]
        final_gen = child[random_pos]
        child[pos_gen] = final_gen
        child[random_pos] = gen
        return child

    def mutate_layer(self, child, pos_gen):
        layer = child[pos_gen].copy()
        if random() < self.mutation_n_neurons:
            layer[0] = round(uniform(*self.num_neurons))
        if random() < self.mutation_f_activation:
            layer[1] = self.function[round(uniform(0, len(self.function) - 1))]
        child[pos_gen] = layer
        return child
    
    def pruning(self):
        pop_list = []
        for p in self.population:
            if p not in pop_list:
                pop_list.append(p)

        self.fitness = [self.calculate_aptitude(x) for x in pop_list]

        pop_sorted = sorted(list(map(lambda x,y:[x,y], self.fitness,pop_list)), reverse=True)
        
        self.population = [x[1] for x in pop_sorted[:self.pop_size]]

    def evaluate(self):
        self.generate_population()
        for generation in range(self.num_generations):
            self.crossover()
            self.mutate()
            self.pruning()
            print("Generation", generation+1, "- Best fitness:", self.fitness[0])

ga = genetic_algorithm(
    num_layers=(3,6),
    num_neurons=(64,256),
    pop_size=10, 
    num_generations=10,
    mut_rate=0.5,
    mut_rate_pos=0.6,
    mut_rate_layer=0.5,
    mut_rate_neurons=0.5,
    mut_rate_f_activation=0.5,
    cross_rate=0.6
)
ga.evaluate()