import NeuroEvolution.geneticAlgorithm as ga
import NeuroEvolution.jsonHandler as jsonH
from RayCasting.boundary import Boundary
from car import Car
import pygame as py
import sys, json, math


# json dosyasından gelen veriler ile bu değşkenler doldurulacak
size = None
boundaries = []
checkpoints = []


# proje içinde 'mapData.json' olduğu varsayılır yoksa hata verir
def load_map_from_json():
    global size, boundaries, checkpoints
    with open('mapData.json', 'r') as json_file:
        data = json.load(json_file)
        size = data['width'], data['height']
        Car.spawnpoint = data['spawnpoint']
        checkpoints = data['checkpoints']
        for w in data['walls']:
            boundaries.append(Boundary(w[0], w[1]))


# diğer değişkenler
py.init()
load_map_from_json()
screen = py.display.set_mode(size)
clock = py.time.Clock()
background = 0, 0, 0

dt = 1
lifeTimeInSecs = 40

generationNum = 1
generationSize = 150
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
    global generationNum, activeCars, timeRemain
    # yeni jenerasyona geçmeden arabalara not verilir
    update_scores()

    activeCars = ga.create_new_generation(oldGeneration=crashedCars)
    crashedCars.clear()

    timeRemain = lifeTimeInSecs
    generationNum += 1


def render():
    screen.fill(background)
    for car in activeCars:
        car.render(screen)
    for b in boundaries:
        b.render(screen)
    for c in checkpoints:
        py.draw.circle(screen, (0, 0, 255), c, 4)
    py.display.flip()


def update():
    update_generation()
    detect_crashes()
    check_checkpoint()

# yardımcı fonksiyolar

def update_generation():
    for car in activeCars:
        car.drive(boundaries, dt)


# daha temiz yazılabilir!!!!
def detect_crashes():
    for car in activeCars:
        # araç harite dışına çıkar ise yanar
        if car.loc[0] < 0 or car.loc[0] > size[0] or car.loc[1] < 0 or car.loc[1] > size[1]:
            activeCars.remove(car)
            crashedCars.append(car)
            continue

        for b in boundaries:
            if car.check_collusion(b):
                activeCars.remove(car)
                crashedCars.append(car)
                break


def update_scores():
    for car in crashedCars:
        car.update_score(len(checkpoints))


def check_checkpoint():
    for car in activeCars:
        try:
            cx = car.loc[0]
            cy = car.loc[1]
            cpx = checkpoints[car.checkpointPassed][0]
            cpy = checkpoints[car.checkpointPassed][1]
        except:
            print('hata')
            save_exit()

        dis = math.sqrt(pow(cx-cpx, 2) + pow(cy-cpy, 2))

        # kontrol noktasını geçti demek
        if dis < 30:
            car.checkpointPassed += 1


def generate_random_gen():
    for i in range(generationSize):
        activeCars.append(Car())


def save_exit():
    update_scores()
    bestCar = pick_best()
    jsonH.save(bestCar.brain, 'bestcar')
    sys.exit()


def save_best():
    bestInd = 0
    for i in range(len(crashedCars)):
        if crashedCars[bestInd].score < crashedCars[i].score:
            bestInd = i
    jsonH.save(crashedCars[bestInd].brain, 'bestcar')


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


def load_gen():
    global activeCars, crashedCars

    brain = jsonH.load_nn('bestcar')
    car = Car(brain=brain)
    car.score = 1
    crashedCars.append(car)
    activeCars = ga.create_new_generation(crashedCars, generationSize)
    crashedCars.clear()


# ana döngü

timeRemain = lifeTimeInSecs

# generate_random_gen()
load_gen()
while 1:
    clock.tick()

    handle_pygame_events()
    update()

    clock.tick()

    dt = clock.get_time() / 1000
    timeRemain -= dt

    # yeni jenerasyona geçilmelimi ona bakılır
    if len(activeCars) == 0:
        save_best()
        setup_generation()
    if timeRemain < 0:
        kill_gen()

    if renderOn is True:
        render()

    py.display.set_caption("Nesil: " + str(generationNum) + '-' + str(len(activeCars)) + '   ' + str(timeRemain))
