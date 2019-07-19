import random
import math
from car import Car


def create_new_generation(oldGeneration, generationlen=None):
    if generationlen is None:
        generationlen = len(oldGeneration)

    calculate_fitness(oldGeneration)
    newGeneration = []

    for i in range(generationlen):
        mom = choose_parent(oldGeneration)
        dad = choose_parent(oldGeneration)
        child = crossover(mom, dad)
        newGeneration.append(child)

    return newGeneration


def calculate_fitness(generation):
    sum = 0
    for member in generation:
        member.fitness = math.pow(member.score, 2)
        sum += member.fitness
    for member in generation:
        member.fitness /= sum


def choose_parent(generation):
    r = random.random()
    index = 0
    while r < 0:
        r += generation[index].score
        index += 1
    index -= 1
    return generation[index]


def crossover(parent1, parent2):
    # beyin nöron ağı objesidir
    brain1 = parent1.brain.copy()
    brain2 = parent2.brain.copy()
    # W: ağırlık F: fitness
    # Wçocuk = (Wbaba * Fbaba + Wana * Fana) / (Fana + Fbaba)
    # aynı kural sapmalar içinde uygulanabilir

    # iki ebebeyn de aynı sayıda nöron ve katmana sahip olduğu için
    # indeks hatası ile uğraşmamız gerekmiyecek
    for i in range(len(brain1.weights)):
        Wp1 = brain1.weights[i]
        Wp2 = brain2.weights[i]
        Bp1 = brain1.biases[i]
        Bp2 = brain2.biases[i]

        # ağırlıkları ve sapmaları ebebeynlerin fitnesslariyla çarparız
        Wp1.multiply(parent1.fitness)
        Wp2.multiply(parent2.fitness)
        Bp1.multiply(parent1.fitness)
        Bp2.multiply(parent2.fitness)

        # işlemlerimizi brain1 üzerinden yapacağız
        # çocuğada brain1 objesini vereceğiz
        Wp1.add(Wp2)
        Wp1.multiply(1 / (parent1.fitness + parent2.fitness))
        Bp1.add(Bp2)
        Bp1.multiply(1 / (parent1.fitness + parent2.fitness))

    brain1.mutate(0.04)

    child = Car(brain=brain1)
    return child
