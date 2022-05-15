#ifndef IMG_OPERATIONS_H
#define IMG_OPERATIONS_H

#include <opencv/cv.h>
#include <opencv/highgui.h>

typedef struct {
    int fila;
    int columna;
    int max_dif;
} block_candidate_t;

/**
 * Recorta una imagen poniendo marcos negros en los bordes
 * @param img
 * @param top Tamaño en píxeles que tendrá el borde superior
 * @param bot Tamaño en píxeles que tendrá el borde inferior
 * @param left Tamaño en píxeles que tendrá el borde izquierdo
 * @param right Tamaño en píxeles que tendrá el borde derecho
 */
void crop_image(IplImage *img, int top, int bot, int left, int right);

void get_candidate_blocks(IplImage* img, block_candidate_t candidates[], int n);

#endif /* IMG_OPERATIONS_H */
