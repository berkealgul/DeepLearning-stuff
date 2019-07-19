from MapTools.Map import Map
import pygame
import sys
import json

# bu dosya harita editörünü çalıştırır similasyondan bağımsızdır
pygame.init()

w = 1200
h = 650

firstClickPos = None
secondClickPos = None

chainMode = False

Map = Map(w, h)
screen = pygame.display.set_mode((w, h))


# fonksiyonlar
def pygame_event_handle():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            save_quit()
        if event.type == pygame.KEYDOWN:
            handle_keyboard_input()
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_mouse_input()


# zaten var olan harita düzenlenmek istenirse bu method ile harita oluşturulmalı
def create_map_from_json():
    with open('mapData.json', 'r') as json_file:
        data = json.load(json_file)
        w = data['width']
        h = data['height']
        sp = data['spawnpoint']
        map = Map(w ,h, sp)
        map.walls = data['walls']
        return map


def save_quit():
    data = Map.convert_to_json()
    with open('mapData.json', 'w') as json_file:
        json.dump(data, json_file)
    sys.exit()


# kullanıcı girdileriyle ilgilenme
def handle_mouse_input():
    pressed = pygame.mouse.get_pressed()
    if pressed[0] == 1:
        left_click()
    elif pressed[2] == 1:
        right_click()
    elif pressed[1] == 1:
        Map.set_spawnpoint(pygame.mouse.get_pos())


def handle_keyboard_input():
    global chainMode, firstClickPos, secondClickPos
    keys = pygame.key.get_pressed()
    if keys[pygame.K_c]:
        if chainMode is False:
            chainMode = True
        else:
            chainMode = False
            firstClickPos = None
    elif keys[pygame.K_TAB]:
        firstClickPos = None
        secondClickPos = None


# duvar ekle çıkarma
def add_wall():
    global firstClickPos, secondClickPos
    Map.add_wall(firstClickPos, secondClickPos)

    if chainMode is False:
        firstClickPos = None
        secondClickPos = None
    else:
        firstClickPos = secondClickPos[0], secondClickPos[1]
        secondClickPos = None


# görüntüleme
def render():
    screen.fill((0, 0, 0))
    Map.render(screen)
    if firstClickPos is not None:
        render_wall_preview()
    pygame.display.flip()


def render_wall_preview():
    global firstClickPos
    pos = pygame.mouse.get_pos()
    pygame.draw.line(screen, (0, 255, 0), firstClickPos, pos, 3)


# fare fonksiyonları
def left_click():
    global firstClickPos, secondClickPos
    pos = pygame.mouse.get_pos()

    if firstClickPos is None:
        firstClickPos = pos
    elif secondClickPos is None:
        secondClickPos = pos
        add_wall()


def right_click():
    pos = pygame.mouse.get_pos()
    w = Map.detect_closest_wall(pos)
    if w is not None:
        Map.remove_wall(w)


# ana döngü
while 1:
    pygame_event_handle()
    render()
    pygame.display.set_caption('sol/sağ tık: duvar ekle/çıkar, '
                               'orta tuş: doğma noktasını ayarla, c: zincir modu, tab: işlem iptali   Zincir mod: '
                               + str(chainMode))
