from random import randint, uniform, random, sample
from itertools import combinations, product
import numpy as np
import sys
sys.path.append("./neuronal_network") 
from neuronal_network import neuronal_network

class genetic_algorithm:
    def __init__(self, num_layers, num_neurons,pop_size,num_generations,mut_rate, mut_rate_pos, mut_rate_layer, mut_rate_neurons,mut_rate_f_activation,num_trains, nn):
        self.num_layers = num_layers 
        self.num_neurons = num_neurons 
        self.function = ['relu','softmax']
        self.pop_initial = randint(2,pop_size)
        self.pop_size = pop_size
        self.num_generations = num_generations
        self.mutation_rate = mut_rate
        self.mutation_pos = mut_rate_pos
        self.mutation_layer = mut_rate_layer
        self.mutation_n_neurons = mut_rate_neurons
        self.mutation_f_activation = mut_rate_f_activation
        self.num_trains = num_trains
        self.population = []
        self.nn = nn
    
    def calculate_aptitude(self, individual):
        print('Calculando aptitud de ', individual)
        num_layers_dense = len(individual)
        num_neurons = [x[0] for x in individual]
        activation = [x[1] for x in individual]

        accurancy_trains=[]

        self.nn.define_model(
                num_layers_dense,
                num_neurons,
                activation
            )

        for i in range(self.num_trains):
            print('Train ',i)
            self.nn.train_model()
            print(self.nn.accuracy_list)
            accurancy_trains.append(np.mean(self.nn.accuracy_list))

        return np.mean(accurancy_trains)

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
        self.fitness = [self.calculate_aptitude(x) for x in self.population]
    
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
            self.fitness.append(self.calculate_aptitude(child))
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
        fitness_list = []

        for i in range(len(self.population)):
            if self.population[i] not in pop_list:
                pop_list.append(self.population[i])
                fitness_list.append(self.fitness[i])

        pop_sorted = sorted(list(map(lambda x,y:[x,y], fitness_list,pop_list)), reverse=True)
        
        self.population = [x[1] for x in pop_sorted[:self.pop_size]]
        self.fitness = [x[0] for x in pop_sorted[:self.pop_size]]

        print("Poblacion final",*pop_sorted,sep='\n')

    def evaluate(self):
        self.generate_population()
        for generation in range(self.num_generations):
            self.crossover()
            self.mutate()
            self.pruning()
            print("Generation", generation+1, "- Best fitness:", self.fitness[0])
        print("mejor individuo", self.population[0])

    def show_result(self):
        #Mostrar resultados
        pass
#Ejecutar AG
nn = neuronal_network()
nn.preprocess_images()
ga = genetic_algorithm(
    num_layers=(2,3),
    num_neurons=(30,70),
    pop_size=5, 
    num_generations=5,
    mut_rate=0.5,
    mut_rate_pos=0.6,
    mut_rate_layer=0.5,
    mut_rate_neurons=0.5,
    mut_rate_f_activation=0.5,
    num_trains=5,
    nn=nn
)

ga.evaluate()