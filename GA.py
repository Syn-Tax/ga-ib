import random
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np
import matplotlib.pyplot as plt
import statistics
import math
# import wandb

distances_large =  [[0,   94,  76,  141, 91,  60,  120, 145, 91,  74,  90,  55,  145, 108, 41,  49,  33,  151, 69,  111, 24  ],
                    [94,  0,   156, 231, 64,  93,  108, 68,  37,  150, 130, 57,  233, 26,  62,  140, 61,  229, 120, 57,  109 ],
                    [76,  156, 0,   80,  167, 133, 124, 216, 137, 114, 154, 100, 141, 161, 116, 37,  100, 169, 49,  185, 84  ],
                    [141, 231, 80,  0,   229, 185, 201, 286, 216, 139, 192, 178, 113, 239, 182, 92,  171, 155, 128, 251, 137 ],
                    [91,  64,  167, 229, 0,   49,  163, 65,  96,  114, 76,  93,  200, 91,  51,  139, 72,  185, 148, 26,  92  ],
                    [60,  93,  133, 185, 49,  0,   165, 115, 112, 65,  39,  91,  151, 117, 39,  99,  61,  139, 128, 75,  49  ],
                    [120, 108, 124, 201, 163, 165, 0,   173, 71,  194, 203, 74,  254, 90,  127, 136, 104, 269, 75,  163, 144 ],
                    [145, 68,  216, 286, 65,  115, 173, 0,   103, 179, 139, 123, 265, 83,  104, 194, 116, 250, 186, 39,  152 ],
                    [91,  37,  137, 216, 96,  112, 71,  103, 0,   160, 151, 39,  236, 25,  75,  130, 61,  239, 95,  93,  112 ],
                    [74,  150, 114, 139, 114, 65,  194, 179, 160, 0,   54,  127, 86,  171, 89,  77,  99,  80,  134, 140, 50  ],
                    [90,  130, 154, 192, 76,  39,  203, 139, 151, 54,  0,   129, 133, 155, 78,  117, 99,  111, 159, 101, 71  ],
                    [55,  57,  100, 178, 93,  91,  74,  123, 39,  127, 129, 0,   199, 61,  53,  91,  30,  206, 63,  101, 78  ],
                    [145, 233, 141, 113, 200, 151, 254, 265, 236, 86,  133, 199, 0,   251, 171, 118, 176, 46,  182, 226, 125 ],
                    [108, 26,  161, 239, 91,  117, 90,  83,  25,  171, 155, 61,  251, 0,   83,  151, 75,  251, 119, 81,  127 ],
                    [41,  62,  116, 182, 51,  39,  127, 104, 75,  89,  78,  53,  171, 83,  0,   90,  24,  168, 99,  69,  49  ],
                    [49,  140, 37,  92,  139, 99,  136, 194, 130, 77,  117, 91,  118, 151, 90,  0,   80,  139, 65,  159, 50  ],
                    [33,  61,  100, 171, 72,  61,  104, 116, 61,  99,  99,  30,  176, 75,  24,  80,  0,   179, 76,  86,  52  ],
                    [151, 229, 169, 155, 185, 139, 269, 250, 239, 80,  111, 206, 46,  251, 168, 139, 179, 0,   202, 211, 128 ],
                    [69,  120, 49,  128, 148, 128, 75,  186, 95,  134, 159, 63,  182, 119, 99,  65,  76,  202, 0,   161, 90  ],
                    [111, 57,  185, 251, 26,  75,  163, 39,  93,  140, 101, 101, 226, 81,  69,  159, 86,  211, 161, 0,   115 ],
                    [24,  109, 84,  137, 92,  49,  144, 152, 112, 50,  71,  78,  125, 127, 49,  50,  52,  128, 90,  115, 0   ]]

distances_small = [i[:8] for i in distances_large[:8]]

distances_medium = [i[:15] for i in distances_large[:15]]

distances = distances_medium

example_tour = range(len(distances[0]))[1::]

class Individual:
  def __init__(self, tour=None):
    if tour:
      self.tour = tour
    else:
      self.tour = random.sample(example_tour, len(example_tour))
    self.fitness = 9e99
    self.probability = 0

  def evaluate(self, func):
    self.fitness = func([0] + self.tour + [0])
  
  def mutate(self, mutation_rate=1e-3):
    for i in range(len(self.tour)):
      if random.random() <= mutation_rate:
        swap = random.randint(0, len(self.tour)-1)
        self.tour[i], self.tour[swap] = self.tour[swap], self.tour[i]


def calculate_distance(tour):
  total = 0
  for i in range(len(tour)-1):
    total += distances[tour[i]][tour[i+1]]
  return 1/total

def initialise_population(pop_size):
  return [Individual() for _ in range(pop_size)]

def truncation_selection(pop, truncation_factor=0.5):
  return sorted(pop, key=lambda x : x.fitness, reverse=True)[:int(len(pop)*truncation_factor+1)]

def tournament_selection(pop, num, probability=1, tournament_size=5):
  individuals = []
  for t in range(num):
    tournament = []
    for _ in range(tournament_size):
      tournament.append(random.choice(pop))

    tournament = sorted(tournament, key=lambda x : x.fitness, reverse=True)

    probabilities = [probability*((1-probability)**i) for i in range(len(tournament))]

    for i in range(len(tournament)):
      if random.random() < probabilities[i]:
        individuals.append(tournament[i])	

  return individuals

def roulette_wheel_selection(population, size=None, wheel=None):
  if not wheel:
    sum_fitness = sum([i.fitness for i in population])
    pop = sorted(population, key=lambda x : x.fitness, reverse=True)
    wheel = []
    total = 0
    for i in pop:
      prob = (i.fitness)/sum_fitness
      wheel.append((i, total+prob))
      total += prob

  if wheel and wheel[-1][1] > 1:
    ValueError("Probabilities do not sum to 1")
  
  if not size:
    size = len(wheel)
    
  chosen = []
  for i in range(size):
    value = random.random()
    if wheel[0][1] > value:
      chosen.append(wheel[0][0])
      
    for j in list(range(len(wheel)))[1::]:
      if wheel[j][1] > value > wheel[j-1][1]:
        chosen.append(wheel[j][0])
        
  return chosen

def stochastic_universal_sampling(population, size, wheel=None):
  if not wheel:
    sum_fitness = sum([i.fitness for i in population])
    pop = sorted(population, key=lambda x : x.fitness, reverse=True)
    wheel = []
    total = 0
    for i in pop:
      prob = (i.fitness)/sum_fitness
      wheel.append((i, total+prob))
      total += prob

  if wheel and wheel[-1][1] > 1:
    ValueError("Probabilities do not sum to 1")
  
  if not size:
    size = len(wheel)
    
  chosen = []
  offset = random.random()
  select_pos = []
  total_pos = offset
  for i in range(size):
    curr_offset = 1/size
    select_pos.append((total_pos + curr_offset) % 1)
    total_pos += curr_offset
    total_pos = total_pos % 1

  for pos in select_pos:
    if wheel[0][1] > pos:
      chosen.append(wheel[0][0])
      
    for j in list(range(len(wheel)))[1::]:
      if wheel[j][1] > pos > wheel[j-1][1]:
        chosen.append(wheel[j][0])

  return chosen

def order_crossover(parent1, parent2):
  p1, p2 = parent1.tour, parent2.tour
  i = random.randint(0, int(len(p1)/2)-1)
  j = random.randint(i+1, int(len(p1))-1)

  f1 = [None for _ in range(len(p1))]

  for k in range(i, j):
    f1[k] = p1[k]

  f1_index = j
  p2_index = j
  while None in f1:
    while p2[p2_index] in f1:
      p2_index += 1
      p2_index = p2_index % len(p2)

    f1[f1_index] = p2[p2_index]
    f1_index += 1
    f1_index = f1_index % len(f1)

  return f1

def partially_mapped_crossover(parent1, parent2):
  p1, p2 = parent1.tour, parent2.tour
  i = random.randint(0, int(len(p1)/2)-1)
  j = random.randint(i+1, int(len(p1))-1)

  f1 = [None for _ in range(len(p1))]

  for k in range(i, j):
    f1[k] = p1[k]

  mappings = {}
  p2_index = 0
  for e in list(range(i))+list(range(j, len(p2))):
    map1 = p1[e]
    while p2[p2_index] in f1:
      p2_index += 1
      p2_index = p2_index % len(p2)
    mappings[map1] = p2[p2_index]
    p2_index += 1
    p2_index = p2_index % len(p2)
  
  for k in range(len(p1)):
    if f1[k] == None:
      f1[k] = mappings[p1[k]]
  
  return f1

def cycle_crossover(parent1, parent2, fill_in=True, random=False):
  p1, p2 = parent1.tour, parent2.tour

  f1 = [None for _ in range(len(p1))]

  if random:
    start = random.randint(0, len(p1)-1)
  else:
    start=0

  f1[start] = p1[start]

  curr_element = p1.index(p2[0])

  while None in f1:
    if not f1[curr_element] == None:
      break
    f1[curr_element] = p1[curr_element]
    curr_element = p1.index(p2[curr_element])
    
  if fill_in:
    for i in range(len(f1)):
      if f1[i] == None:
        f1[i] = p2[i]

  return f1

def main():
  GENERATIONS = 200
  POP_SIZE = 1000

  TRUNCATION_FACTOR = 0.5

  TOURNAMENT_PROBABILITY = 1
  TOURNAMENT_SIZE = 5
  TOURNAMENT_NUMBER = int(3*(POP_SIZE/TOURNAMENT_SIZE))

  ROULETTE_WHEEL_SIZE = int(POP_SIZE/2)

  SUS_SIZE = int(POP_SIZE/2)

  CONVERGENCE_PARAM = 0

  INIT_MUTATION_RATE = 1e-3
  MUTATION_RATE = INIT_MUTATION_RATE
  MUTATION_DECAY = False
  MUTATION_DECAY_RATE = 1e-6

  ADD_RANDOM = True

  CROSSOVER = cycle_crossover

  population = initialise_population(POP_SIZE)
  best_individuals = []
  avg_fitness = []
  stddev = []

  # wandb.init(project="genetic-algorithm")
  break_next = False

  for generation in range(GENERATIONS):
  # while not break_next:
    [i.evaluate(calculate_distance) for i in population]

    min_distance = int(min([1/i.fitness for i in population]))
    avg_distance = int(sum([1/i.fitness for i in population])/len(population))
    stddev_pop = statistics.pstdev([1/i.fitness for i in population])

    print(f"{generation}: {min_distance}: {avg_distance}")
    best_individuals.append(min_distance)
    avg_fitness.append(avg_distance)
    stddev.append(stddev_pop)

    # wandb.log({"min distance": min_distance, "avg distance": avg_distance, "std dev": stddev_pop})

    if break_next:
      break

    if abs(min_distance-avg_distance) <= CONVERGENCE_PARAM:
      break_next = True

    # mating_pool = truncation_selection(population, truncation_factor=TRUNCATION_FACTOR) # TRUNCATION SELECTION
    mating_pool = tournament_selection(population, TOURNAMENT_NUMBER, TOURNAMENT_PROBABILITY, TOURNAMENT_SIZE) # TOURNAMENT SELECTION
    # mating_pool = roulette_wheel_selection(population, size=ROULETTE_WHEEL_SIZE) # ROULETTE WHEEL SELECTION
    # mating_pool = stochastic_universal_sampling(population, SUS_SIZE) # STOCHASTIC UNIVERSAL SAMPLING

    random.shuffle(mating_pool)

    new_population = []

    for i in range(len(mating_pool)-1):
      new_population.append(Individual(tour=CROSSOVER(mating_pool[i], mating_pool[i+1])))
      new_population.append(Individual(tour=CROSSOVER(mating_pool[i+1], mating_pool[i])))

    if not ADD_RANDOM:
      new_population.append(Individual(tour=CROSSOVER(mating_pool[0], mating_pool[-1])))
      new_population.append(Individual(tour=CROSSOVER(mating_pool[1], mating_pool[-2])))

    while len(new_population) < POP_SIZE:
      new_population.append(Individual())
    
    if len(new_population) > POP_SIZE:
      new_population = new_population[:POP_SIZE]
    
    if MUTATION_DECAY:
      MUTATION_RATE = INIT_MUTATION_RATE*math.exp(-MUTATION_DECAY_RATE*generation)

    [i.mutate(mutation_rate=MUTATION_RATE) for i in new_population]

    population = new_population


  [i.evaluate(calculate_distance) for i in population]

  distance_matrix = np.array(distances)
  best_tour, best_dist = solve_tsp_simulated_annealing(distance_matrix)
  print(f"{best_tour}: {best_dist}")
  print(int(1/calculate_distance(best_tour + [0])))
  plt.plot(best_individuals)
  plt.plot(avg_fitness)
  plt.plot([best_dist for _ in range(len(avg_fitness))])
  plt.show()

if __name__ == "__main__":
  main()

# parent1 = Individual()
# parent2 = Individual()
# print(parent1.tour)
# print(parent2.tour)

# cycle_crossover(parent1, parent2, fill_in=False)