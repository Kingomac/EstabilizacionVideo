#ifndef BLOCK_OPERATIONS_H
#define BLOCK_OPERATIONS_H
#define BLOCK_SIZE 32

/**
 * Compara dos bloques
 * @param i fila del píxel esquina del bloque A
 * @param j columna del pixel esquina del bloque A
 * @param imgA imagen A
 * @param k fila del píxel esquina del bloque B
 * @param l columna del pixel esquina del bloque B
 * @param imgB imagen B
 * @return diferencia entre los dos bloques siendo mínimo 0 y máximo 255 * nChannels * BLOCK_SIZE * BLOCK_SIZE
 */
int block_compare(int i, int j, IplImage *imgA, int k, int l, IplImage *imgB);

/**
 * Copia un bloque de una imagen A en una imagen B
 * @param i fila píxel esquina del bloque origen
 * @param j columna píxel esquina del bloque origen
 * @param imgOri imagen de origen
 * @param k fila píxel esquina del bloque destino
 * @param l columna píxel esquina del bloque destino
 * @param imgDest imagen de destino
 */
void block_copy(int i, int j, IplImage *imgOri, int k, int l, IplImage *imgDest);

/**
 * Calcula la intensidad de un microbloque (la imagen debe estar en escala de grises)
 * @param img imagene n escala de grises
 * @param i fila píxel esquina del bloque origen
 * @param j columna píxel esquina del bloque origen
 * @return intensidad = suma de todos los píxeles
 */
int block_intensity(IplImage* img, int i, int j);

#endif /* BLOCK_OPERATIONS_H */
