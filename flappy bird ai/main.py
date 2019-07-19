import pygame
import sys
from NeuroEvolution.matrix import Matrix
import NeuroEvolution.geneticAlgorithm as ga
from birdFile import Bird
from columnFile import Column

# oyuna hazırlık
pygame.init()

# başlangıç için gerekli değişkenler
# jenerasyon sonrasi değişmezler
screenH = 500
screenW = 620
size = (screenW, screenH)
screen = pygame.display.set_mode(size)
clock = pygame.time.Clock()

FPS = 30
deltaTime = FPS / 100
offsetBetweenColumns = 300
Bird.startLoc = (50, screenH / 2)
Bird.dimensions = (32, 20)
population = 500

speed = 1

# her jenerasyon başı yenilenmeleri gereken değişkenler
columns = list()
closestColumn = None

generationNumber = 0
survivingAgents = list()
deadAgents = list()


################################################################################


# similasyonu yürüten öncül fonksiyonlar
def setup_environment():
    global survivingAgents
    global closestColumn
    global generationNumber

    # ilk nesilde rastgele oluştururuz
    if generationNumber == 0:
        for i in range(population):
            survivingAgents.append(Bird(None))
    else:
        survivingAgents = ga.create_new_generation(deadAgents)

    deadAgents.clear()

    columns.clear()
    columns.append(Column(300, screenH))
    columns.append(Column(300 + offsetBetweenColumns, screenH))
    closestColumn = columns[0]

    generationNumber += 1
    speed = 1


def update_columns():
    for col in columns:
        col.move(deltaTime)
        if col.upRect.x < -col.width - 5:
            columns.remove(col)
            # eğer kolon artık göremeyeceğimiz yerdeyse siliyoruz


def update_birds():
    global closestColumn

    for agent in survivingAgents:
        agent.move(deltaTime)
        input = get_input_for_bird(agent)
        agent.think(input)
        check_if_bird_is_dead(agent)

        # kuş en yakın kolondan geçerse(ortasından) yeni kolon oluşturup en yakınını ayarlıyoruz
        if agent.rect.x > closestColumn.upRect.x + closestColumn.width / 2:
            closestColumn = set_closest_column()

        update_score(agent)


def render_screen():
    screen.fill((0, 0, 0))  # arkaplan rengini ayarlıyoruz
    for agent in survivingAgents:
        agent.render(screen)
    for col in columns:
        col.render(screen)
    pygame.display.flip()


################################################################################


# öncül fonksiyonlara yardımcı alt fonksiyonlar
def kill_bird(bird):
    survivingAgents.remove(bird)
    deadAgents.append(bird)


def check_if_bird_is_dead(bird):
    if bird.rect.y > screenH or bird.rect.y < 0:
        kill_bird(bird)
    # yere veya tavana çakıldımı ona bakıyoruz

    if closestColumn.is_hit_to_bird(bird):
        kill_bird(bird)
    # en yakın kolon oyuncuya vurdumu ona bakıyoruz


def get_input_for_bird(bird):
    input = Matrix(5, 1)
    # input =
    # [ closestColumn.upRect.x / screenW ]
    # [ closestColumn.lowerLimitOfUpRect / screenH ]
    # [ closestColumn.downRect.y / screenH ]
    # [ bird.rect.y / screenH ]
    # [ bird.velocity.y ]
    input.values[0][0] = closestColumn.upRect.x / screenW
    input.values[1][0] = closestColumn.lowerLimitOfUpRect / screenH
    input.values[2][0] = closestColumn.downRect.x / screenH
    input.values[3][0] = bird.rect.y / screenH
    input.values[4][0] = bird.velocityY / bird.maxSpeed
    return input


def update_score(bird):
    # hayatta kaldığı için normal score artışı
    bird.score += 1

    # eğer kuş yüksekliğini sütunların arasında tutabilmiş ise fazladan puan kazanır
    if closestColumn.lowerLimitOfUpRect < bird.rect.y < closestColumn.downRect.y:
        bird.score += 3


def spawn_column():
    locXofNewColumn = columns[len(columns) - 1].upRect.x + offsetBetweenColumns
    columns.append(Column(locXofNewColumn, screenH))


def set_closest_column():
    spawn_column()
    return columns[1]


################################################################################


# ana similasyon döngüsü
setup_environment()
while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()
        if event.type == pygame.MOUSEBUTTONUP:
            if pygame.mouse.get_pressed()[0]:
                speed += 1
            elif pygame.mouse.get_pressed()[1]:
                speed -= 1

    for i in range(speed):
        update_columns()
        update_birds()
        render_screen()

    if len(survivingAgents) < 20:
        speed = 5

    # bütün herkez öldü ise bi sonraki nesil hazırlanır
    if len(survivingAgents) == 0:
        setup_environment()

    clock.tick(FPS)
    pygame.display.set_caption("Nesil " + str(generationNumber))
