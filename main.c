
#include <stdio.h>
#include <stdlib.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <emmintrin.h>

#include "img_operations.h"
#include "block_operations.h"


#define OFFSET 20
#define NTHREADS 4

typedef struct {
    int i;
    int j;
} vec_t;

/**
 * Busca un bloque a su alrededor en una imagen llegando como máximo al OFFSET
 * @param row Fila de la esquina superior izquierda del bloque
 * @param col Columna de la esquina superior izquierda del bloque
 * @param prevBlock Imagen que contiene el bloque a buscar
 * @param frame Frame actual
 * @return Vector con el desplazamiento de bloque
 */
vec_t look_for_block(const int row, const int col, IplImage* prevBlock, IplImage* frame) {
    int min_dif = INT_MAX;
    vec_t res = (vec_t){0, 0};
    for (int i = -OFFSET; i <= OFFSET; i++) {
        for (int j = -OFFSET; j <= OFFSET; j++) {
            int dif = block_compare(0, 0, prevBlock, row + i, col + j, frame);
            if (dif == 0) {
                return (vec_t){ i, j};
            }
            if (dif < min_dif) {
                min_dif = dif;
                res = (vec_t){i, j};
            }
        }
    }
#ifdef DEBUG
    printf("Block not found\n");
#endif
    return res;
}

struct CandidateCalculateArgs {
    block_candidate_t* block;
    int cropY;
    int cropX;
    IplImage* frame;
};

void calculate_candidate(void* args) {
    struct CandidateCalculateArgs* data = (struct CandidateCalculateArgs*) args;

    const vec_t dir = look_for_block(data->block->fila + data->cropY, data->block->columna + data->cropX, data->block->block, data->frame);
    data->cropY = dir.i;
    data->cropX = dir.j;
}

int main(int argc, char** argv) {


    if ((argc != 2 && argc != 3) || argc == 3 && strcmp(argv[2], "-showoff") != 0) {
        printf("Argumentos incorrectos\nestabilicacionvideo [ruta archivo de vídeo] [-showoff]\n");
        exit(-1);
    }

    struct timespec start, finish;
    double elapsed;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // Creamos las imagenes a mostrar
    CvCapture* capture = cvCaptureFromAVI(argv[1]);

    // Always check if the program can find a file
    if (!capture) {
        printf("Error: fichero %s no leido\n", argv[1]);
        return EXIT_FAILURE;
    }

#ifdef DEBUG
    printf("Modo Debug\n");
#endif

    IplImage *frame = cvQueryFrame(capture);
    IplImage *outputFrame = cvCloneImage(frame);

    IplImage* primerFrame = cvCloneImage(frame);
    vec_t crop = (vec_t){0, 0};

    block_candidate_t candidates[NTHREADS];

    get_candidate_blocks(frame, candidates, NTHREADS);

    /*cvNamedWindow("a", CV_WINDOW_KEEPRATIO);
    cvNamedWindow("b", CV_WINDOW_KEEPRATIO);
    for (int i = 0; i < NTHREADS; i++) {
        cvShowImage("a", candidates[i].block);
        IplImage* xd = cvCloneImage(frame);
        cvRectangle(xd, cvPoint(candidates[i].columna, candidates[i].fila), cvPoint(candidates[i].columna + BLOCK_SIZE, candidates[i].fila + BLOCK_SIZE), cvScalar(0, 0, 255, 0), 1, LINE_MAX, 0);
        cvShowImage("b", xd);
        cvWaitKey(0);
    }

    printf("%d\n\n\n", block_compare(0, 0, candidates[0].block, candidates[0].fila, candidates[0].columna, frame));
    printf("%d\n\n\n", block_compare(224, 373, frame, candidates[0].fila, candidates[0].columna, frame));*/


    pthread_t threads[NTHREADS];
    struct CandidateCalculateArgs args[NTHREADS];

    do {

        for (int i = 0; i < NTHREADS; i++) {
            args[i] = (struct CandidateCalculateArgs){
                &candidates[i],
                crop.i,
                crop.j,
                frame,
            };
            pthread_create(&threads[i], NULL, (void*) &calculate_candidate, (void*) &args[i]);
        }

        int sum_cropX = 0;
        int sum_cropY = 0;

        for (int i = 0; i < NTHREADS; i++) {
            pthread_join(threads[i], NULL);
            sum_cropY += args[i].cropY;
            sum_cropX += args[i].cropX;
        }

        crop.i += sum_cropY / NTHREADS;
        crop.j += sum_cropX / NTHREADS;

#ifdef DEBUG
        printf("cropY: %d, cropX: %d\n", crop.i, crop.j);
#endif

        if (argc != 3) {
            int cropTop = crop.i < 0 ? -crop.i : 0;
            int cropBot = crop.i > 0 ? crop.i : 0;
            int cropLeft = crop.j < 0 ? -crop.j : 0;
            int cropRight = crop.j > 0 ? crop.j : 0;
            crop_image(outputFrame, cropTop, cropBot, cropLeft, cropRight);
            cvShowImage("Resultado", outputFrame);
            cvShowImage("Original", frame);
            cvWaitKey(0);
        }
        cvCopy(primerFrame, outputFrame, NULL);
    } while ((frame = cvQueryFrame(capture)) != NULL);

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = finish.tv_sec - start.tv_sec;
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Tiempo de ejecución: %f\n", elapsed);
}
