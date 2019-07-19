import NeuroEvolution.geneticAlgorithm as ga
import NeuroEvolution.jsonHandler as jsonH
from RayCasting.boundary import Boundary
from car import Car
import pygame as py
import sys, json


# json dosyasından gelen veriler ile bu değşkenler doldurulacak
size = None
boundaries = []


# proje içinde 'mapData.json' olduğu varsayılır yoksa hata verir
def load_map_from_json():
    global size, boundaries
    with open('mapData.json', 'r') as json_file:
        data = json.load(json_file)
        size = data['width'], data['height']
        Car.spawnpoint = data['spawnpoint']
        for w in data['walls']:
            boundaries.append(Boundary(w[0], w[1]))


# diğer değişkenler
py.init()
load_map_from_json()
screen = py.display.set_mode(size)
clock = py.time.Clock()
background = 0, 0, 0
FPS = 60
dt = 1 / FPS

generationNum = 1
generationSize = 80
activeCars = []
crashedCars = []

renderOn = True


# döngüyü sağlayacak fonksiyonlar
def handle_pygame_events():
    for event in py.event.get():
        if event.type == py.QUIT:
            save_exit()
        if event.type == py.MOUSEBUTTONDOWN:
            pressed = py.mouse.get_pressed()
            if pressed[0] == 1:
                kill_gen()
            elif pressed[2] == 1:
                toggle_render_mod()


def setup_generation():
    global generationNum, activeCars
    activeCars = ga.create_new_generation(crashedCars)
    crashedCars.clear()
    generationNum += 1


def render():
    screen.fill(background)
    for car in activeCars:
        car.render(screen)
    for b in boundaries:
        b.render(screen)
    py.display.flip()


def update():
    update_generation()
    detect_crashes()
    update_scores()


# yardımcı fonksiyolar
def update_generation():
    for car in activeCars:
        car.drive(boundaries, dt)


def detect_crashes():
    for car in activeCars:
        for b in boundaries:
            if car.check_collusion(b):
                activeCars.remove(car)
                crashedCars.append(car)
                break


# ! optimize edilmeli
def update_scores():
    for car in activeCars:
        incremant = (car.velocity / car.maxSpeed)
        car.score += incremant * dt


def generate_random_gen():
    for i in range(generationSize):
        activeCars.append(Car())


def save_exit():
    bestCar = pick_best()
    jsonH.save(bestCar.brain, 'bestcar')
    sys.exit()


def pick_best():
    bestInd = 0
    for i in range(len(activeCars)):
        if activeCars[bestInd].score < activeCars[i].score:
            bestInd = i
    return activeCars[bestInd]


def kill_gen():
    for car in activeCars:
        crashedCars.append(car)
    activeCars.clear()


def toggle_render_mod():
    global renderOn
    if renderOn is True:
        renderOn = False
        screen.fill(background)
        py.display.flip()
    else:
        renderOn = True


# ana döngü
#generate_random_gen()
brain = jsonH.load_nn('bestcar')
car = Car(brain=brain)
car.score = 1
crashedCars.append(car)
activeCars = ga.create_new_generation(crashedCars, generationSize)
crashedCars.clear()

while 1:
    clock.tick()

    handle_pygame_events()
    update()

    if renderOn is True:
        render()

    if len(activeCars) == 0:
        setup_generation()

    clock.tick()
    dt = clock.get_time() / 1000

    bestCar = pick_best()
    py.display.set_caption("Nesil: " + str(generationNum) + ' en yüksek skor: '+ str(bestCar.score))

