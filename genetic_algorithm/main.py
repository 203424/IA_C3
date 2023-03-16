from random import randint, uniform, random, sample
from itertools import combinations, product

class genetic_algorithm:
    def __init__(self, num_layers, num_neurons,pop_size,num_generations,mut_rate, cross_rate):
        self.num_layers = num_layers #rango ej. (3,6)
        self.num_neurons = num_neurons #rango ej. (64,256)
        self.function = ['relu','softmax','sigmoide']
        self.pop_initial = randint(2,pop_size)
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mut_rate
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
        for child in range(len(self.children)):
            mutate_child = self.children[child].copy()
            if random() < self.mutation_rate:
                for pos_gen in range(len(mutate_child)):
                    if random() < self.mutation_rate:
                        mutate_child[pos_gen] = [round(uniform(*self.num_neurons)),self.function[round(uniform(0,len(self.function)-1))]]
                self.children.pop(child)
                self.children.insert(child, mutate_child)

        self.population.extend(self.children)
    
    def pruning(self):
        pop_list = []
        #se eliminan repetidos
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
    pop_size=5, 
    num_generations=100,
    mut_rate=0.5,
    cross_rate=0.6
)
ga.evaluate()