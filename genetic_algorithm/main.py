from random import randint, uniform, random, sample
from itertools import combinations, product
import numpy as np
from tkinter import Tk,Frame,Label, Button, Entry
import matplotlib.pyplot as plt
import sys, os, shutil
sys.path.append("./neuronal_network") 
from neuronal_network import neuronal_network

class genetic_algorithm:
    def __init__(self, num_layers, num_neurons,pop_size,num_generations,mut_rate, mut_rate_pos, mut_rate_layer, mut_rate_neurons,mut_rate_f_activation, nn):
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
        self.population = []
        self.nn = nn
        #results
        self.generations = []
        self.accurancies_list = []
        self.best_fitness = []
        self.avg_fitness = []
        self.worst_fitness = []
    
    def calculate_aptitude(self, individual):
        num_layers_dense = len(individual)
        num_neurons = [x[0] for x in individual]
        activation = [x[1] for x in individual]

        self.nn.define_model(
                num_layers_dense,
                num_neurons,
                activation
            )

        self.nn.train_model(epochs=20)

        # print(f'{individual}, - fit: {np.mean(self.nn.accuracy_list)}')

        self.accurancies_list.append(np.mean(self.nn.accuracy_list))

        return np.mean(self.nn.accuracy_list) #precision media del modelo

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
            print("cr_m",crosspoint_max)
            if crosspoint_max != 0:
                for i in range(2):
                    crosspoint_max = randint(1,crosspoint_max)
                    print("cr_m_new",crosspoint_max)
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
            else:
                #cuando el individuo solo tiene 1 capa, los puntos de cruza maximos es igual a 0
                #Determinar que padre tiene 1 capa
                if len(p[0]) == 1:
                    #definir la capa a intercambiar
                    layer = randint(0,len(p[1])) - 1
                    #se intercambia para crear al nuevo hijo
                    child_2 = p[1].copy()
                    child_1 = child_2.pop(layer)
                    child_2.insert(layer,*p[0]) 
                elif len(p[1]) == 1:
                    #definir la capa a intercambiar
                    layer = randint(0,len(p[1])) - 1
                    #se intercambia para crear al nuevo hijo
                    child_2 = p[0].copy()
                    child_1 = child_2.pop(layer)
                    child_2.insert(layer,*p[1]) 
            self.children += [child_1, child_2]

    def mutate(self):
        for child in self.children:
            if random() < self.mutation_rate:
                child = self.mutate_child(child)
            print(child)
            self.fitness.append(self.calculate_aptitude(child))
            self.population.append(child)

    def mutate_child(self, child):
        mutate_child = child.copy()
        for pos_gen in range(len(mutate_child)):
            if random() < self.mutation_pos:
                mutate_child = self.mutate_position(mutate_child, pos_gen)
            if random() < self.mutation_layer:
                if len(mutate_child) != 1:
                    mutate_child = self.mutate_layer(mutate_child, pos_gen)
                elif len(mutate_child) == 1:
                    mutate_child = self.mutate_layer(mutate_child,-1)
        return mutate_child

    def mutate_position(self, child, pos_gen):
        random_pos = round(uniform(0, len(child) - 1))
        gen = child[pos_gen]
        final_gen = child[random_pos]
        child[pos_gen] = final_gen
        child[random_pos] = gen
        return child

    def mutate_layer(self, child, pos_gen):
        if pos_gen != -1:
            layer = child[pos_gen].copy()
        elif pos_gen == -1:
            layer = child[0]
        if random() < self.mutation_n_neurons:
            layer[0] = round(uniform(*self.num_neurons))
        if random() < self.mutation_f_activation:
            layer[1] = self.function[round(uniform(0, len(self.function) - 1))]
        child[pos_gen] = layer
        return child
    
    def pruning(self):
        pop_list = []
        fitness_list = []
        aux_accurancies_list = []

        for i in range(len(self.population)):
            if self.population[i] not in pop_list:
                pop_list.append(self.population[i])
                fitness_list.append(self.fitness[i])
                aux_accurancies_list.append(self.accurancies_list[i])

        pop_sorted = sorted(list(map(lambda x,y,z:[x,y,z], fitness_list,pop_list,aux_accurancies_list)), reverse=True)
        
        self.best_fitness.append(pop_sorted[0][0])
        self.avg_fitness.append(pop_sorted[0][int(len(pop_sorted)/2)])
        self.worst_fitness.append(pop_sorted[0][len(pop_sorted)])
        # print('pop sorted: ',*pop_sorted,sep='\n')

        self.fitness = [x[0] for x in pop_sorted[:self.pop_size]]
        self.population = [x[1] for x in pop_sorted[:self.pop_size]]
        self.accurancies_list = [x[2] for x in pop_sorted[:self.pop_size]]

    def evaluate(self):
        self.generate_population()
        self.generations.append([self.population.copy(),self.accurancies_list.copy()]) #generacion 0
        for generation in range(self.num_generations):
            self.crossover()
            self.mutate()
            self.pruning()
            self.generations.append([self.population.copy(),self.accurancies_list.copy()])
            print("Generation", generation+1, "- Best fitness:", self.fitness[0])
        print("mejor individuo", self.population[0])
        #Se hace un ultimo entrenamiento de la red con el mejor modelo
        num_layers_dense = len(self.population[0])
        num_neurons = [x[0] for x in self.population[0]]
        activation = [x[1] for x in self.population[0]]
        self.nn.define_model(num_layers_dense,num_neurons,activation)
        self.nn.train_model(epochs=20)
        self.nn.save_model() 
        mvp['text'] = "Mejor modelo: " + str(self.population[0])
        accurancy['text'] = "Precisión: " + str(self.fitness[0])
        self.show_result()

    def generate_tables(self):
        for i in range(len(self.generations)):
            fig, ax = plt.subplots()
            columns_lbl = ('ID', 'Individuo',
                            'Precision media (aptitud)',)
            row = []
            individual,accurancy  =  self.generations[i]
            for j in range(len(individual)):
                row.append([j,individual[j], accurancy[j]])
            ax.set_title("Tabla generación " + str(i))
            ax.axis('off')
            ax.table(
                cellText=row,
                colLabels=columns_lbl,
                loc='center',
            )
            if(i == 0):
                plt.savefig("./genetic_algorithm/graficas/Tabla generación 0 (poblacion inicial).png", dpi=1080,
                        transparent=False,
                        bbox_inches='tight')
            else:
                plt.savefig("./genetic_algorithm/graficas/Tabla generación " + str(i)+".png", dpi=1080,
                            transparent=False,bbox_inches='tight')
            plt.close()

    def graph_best_individual(self):
        fig,ax = plt.subplots(dpi=90, figsize=(10,5))
        plt.suptitle('Evolución de los individuos')
        ax.set_xlabel('Generacion')
        ax.set_ylabel('Aptitud')
        x_values = np.arange(self.num_generations)

        ax.plot(x_values, self.worst_fitness,marker='o',linestyle='dashed', color='r', label='Peor')
        ax.plot(x_values, self.avg_fitness,marker='o',linestyle='dashed', color='g', label='Media')
        ax.plot(x_values, self.best_fitness,marker='o',linestyle='dashed', color='b', label='Mejor')
        ax.legend()

        plt.grid()

        plt.savefig("./genetic_algorithm/graficas/Evolución de los individuos.png", dpi=1080,
                            transparent=False,bbox_inches='tight')

        plt.show(block=False)

    def show_result(self):
        target_dir = './genetic_algorithm/graficas/'
        
        if not os.path.exists(target_dir):
            os.mkdir(target_dir)
        else:
            shutil.rmtree(target_dir, ignore_errors=False, onerror=None)

            os.mkdir(target_dir)

        self.generate_tables()
        self.graph_best_individual()

def iniciar():
    aux = entry_num_layers.get().split(",",2)
    num_layers = (int(aux[0]),int(aux[1]))
    aux = entry_num_neurons.get().split(",",2)
    num_neurons = (int(aux[0]),int(aux[1]))
    pop_size = int(entry_pop_size.get())
    num_generations = int(entry_generaciones.get())
    mut_rate = float(entry_mut_rate.get())
    mut_rate_pos = float(entry_mut_rate_pos.get())
    mut_rate_layer = float(entry_mut_rate_layer.get())
    mut_rate_neurons = float(entry_mut_rate_neurons.get())
    mut_rate_f_activation = float(entry_mut_rate_f_activation.get())
    folds = int(entry_folds.get())
    #Ejecutar AG
    nn = neuronal_network()
    nn.preprocess_images(folds)
    ga = genetic_algorithm(
        num_layers, 
        num_neurons, 
        pop_size, 
        num_generations, 
        mut_rate, 
        mut_rate_pos, 
        mut_rate_layer, 
        mut_rate_neurons, 
        mut_rate_f_activation,
        nn=nn,
    )
    ga.evaluate()

'''Interface'''
tk = Tk()
tk.geometry('750x250')
tk.wm_title('Algoritmo genético para determinar la mejor arquitectura de una red neuronal convolucional')
tk.minsize(width=800, height=600)
tk.config(background='#fefffe')

form_frame = Frame(tk,background='#fefffe',bd=3)
form_frame.grid(column=0, row=0)

mvp = Label(form_frame, text="Mejor modelo: ???", font='Arial 20')
mvp.grid(column=0, row=11, padx=5, pady=5, sticky="EW")
accurancy = Label(form_frame, text="Precisión: ???", font='Arial 20')
accurancy.grid(column=0, row=12, padx=5, pady=5, sticky="EW")

font_lbl = 'Arial 14'

#generaciones
Label(form_frame, font=font_lbl, text='Generaciones: ').grid(column=0, row=0, sticky='E', padx=5, pady=5)
entry_generaciones = Entry(form_frame,  font=font_lbl)
entry_generaciones.grid(column=1, row=0, sticky='W', padx=5, pady=5)
#pop_size
Label(form_frame, font=font_lbl, text='Población máxima: ').grid(column=0, row=1, sticky='E', padx=5, pady=5)
entry_pop_size = Entry(form_frame,  font=font_lbl)
entry_pop_size.grid(column=1, row=1, sticky='W', padx=5, pady=5)
#num_layers
Label(form_frame, font=font_lbl, text='Numero de capas (rango): ').grid(column=0, row=2, sticky='E', padx=5, pady=5)
entry_num_layers = Entry(form_frame,  font=font_lbl)
entry_num_layers.grid(column=1, row=2, sticky='W', padx=5, pady=5)
#num_neurons
Label(form_frame, font=font_lbl, text='Numero de neuronas (rango): ').grid(column=0, row=3, sticky='E', padx=5, pady=5)
entry_num_neurons = Entry(form_frame,  font=font_lbl)
entry_num_neurons.grid(column=1, row=3, sticky='W', padx=5, pady=5)
#Probabilidad de mutacion
Label(form_frame, font=font_lbl, text='Probabilidad de mutación: ').grid(column=0, row=4, sticky='E', padx=5, pady=5)
entry_mut_rate = Entry(form_frame,  font=font_lbl)
entry_mut_rate.grid(column=1, row=4, sticky='W', padx=5, pady=5)
#Probabilidad de mutacion (pos)
Label(form_frame, font=font_lbl, text='Probabilidad de mutar la posición: ').grid(column=0, row=5, sticky='E', padx=5, pady=5)
entry_mut_rate_pos = Entry(form_frame,  font=font_lbl)
entry_mut_rate_pos.grid(column=1, row=5, sticky='W', padx=5, pady=5)
#Probabilidad de mutacion (gen)
Label(form_frame, font=font_lbl, text='Probabilidad de mutar la capa: ').grid(column=0, row=6, sticky='E', padx=5, pady=5)
entry_mut_rate_layer = Entry(form_frame,  font=font_lbl)
entry_mut_rate_layer.grid(column=1, row=6, sticky='W', padx=5, pady=5)
#Probabilidad de mutacion (num_neurons)
Label(form_frame, font=font_lbl, text='Probabilidad de mutar el numero de neuronas: ').grid(column=0, row=7, sticky='E', padx=5, pady=5)
entry_mut_rate_neurons = Entry(form_frame,  font=font_lbl)
entry_mut_rate_neurons.grid(column=1, row=7, sticky='W', padx=5, pady=5)
#Probabilidad de mutacion (f_activation)
Label(form_frame, font=font_lbl, text='Probabilidad de mutar la función de activacion: ').grid(column=0, row=8, sticky='E', padx=5, pady=5)
entry_mut_rate_f_activation = Entry(form_frame,  font=font_lbl)
entry_mut_rate_f_activation.grid(column=1, row=8, sticky='W', padx=5, pady=5)
#Numero de entrenamientos (MonteCarlo)
Label(form_frame, font=font_lbl, text='Cantidad de Folds: ').grid(column=0, row=9, sticky='E', padx=5, pady=5)
entry_folds = Entry(form_frame,  font=font_lbl)
entry_folds.grid(column=1, row=9, sticky='W', padx=5, pady=5)

Button(form_frame,font=font_lbl, text='Iniciar', width=15, bg='#D580FF',fg='white', command=iniciar).grid(column=0, row=10, sticky='EW', pady=5, padx=8, columnspan=2)

tk.mainloop()