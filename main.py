import pygame
import os
import numpy as np
import neuralnetwork
import tensorflow as tf
pygame.font.init()

WIDTH, HEIGHT = 1200, 900
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("MNIST Digit Classifier")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

RESET = (130, 700, 200, 70)
PREDICT = (430, 700, 200, 70)

FONT = pygame.font.SysFont('arial', 50)

FPS = 300

imgtensor = np.zeros((28, 28))
prednum = -1

model = neuralnetwork.model

def draw_window():
    WIN.fill(WHITE)
    curx = 100
    cury = 100
    for row in imgtensor:
        for cell in row:
            pygame.draw.rect(WIN, (cell, cell, cell), (curx, cury, 20, 20))
            curx = curx + 20
        curx = 100
        cury = cury + 20
    pygame.draw.rect(WIN, BLACK, RESET)
    pygame.draw.rect(WIN, BLACK, PREDICT)
    
    reset_font = FONT.render("RESET", 1, WHITE)
    WIN.blit(reset_font, (160, 708))
    predict_font = FONT.render("PREDICT", 1, WHITE)
    WIN.blit(predict_font, (440, 708))
    prediction_font = FONT.render("Prectiction:", 1, BLACK)
    WIN.blit(prediction_font, (800, 200))
    if prednum > -1:
        guess_font = FONT.render(str(prednum), 1, BLACK)
        WIN.blit(guess_font, (900, 300))
    pygame.display.update()


def main():
    global imgtensor
    global prednum
    click = False
    clock = pygame.time.Clock()
    run = True
    while(run):
        clock.tick(FPS)
        mx, my = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                click = True
                if mx >= 130 and mx <= 330 and my >= 700 and my <= 770:
                    imgtensor = np.zeros((28, 28))
                if mx >= 430 and mx <= 630 and my >= 700 and my <= 770:
                    pred = model.predict(np.array([imgtensor]))
                    prednum = np.argmax(pred)
                
            if event.type == pygame.MOUSEBUTTONUP:
                click = False
        if (click):
            x = int((mx-100)/20)
            y = int((my-100)/20)
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = 255
            x = x - 1
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            x = x + 2
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            y = y - 1
            x = x - 1
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            y = y + 2
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            y = y - 2
            x = x - 1
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            x = x + 2
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            y = y + 2
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
            x = x - 2
            if x >= 0 and x < 28 and y >= 0 and y < 28:
                imgtensor[y][x] = max(imgtensor[y][x], 128)
        draw_window()
                
    pygame.quit()
    
if __name__ == "__main__":
    main()