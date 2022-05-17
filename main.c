
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
 * @param prevFrame Frame previo
 * @param frame Frame actual
 * @return Vector con el desplazamiento de bloque
 */
vec_t look_for_block(const int row, const int col, IplImage* prevFrame, IplImage* frame) {
    int min_dif = INT_MAX;
    vec_t res = (vec_t){0, 0};
    for (int i = -OFFSET; i <= OFFSET; i++) {
        for (int j = -OFFSET; j <= OFFSET; j++) {
            int dif = block_compare(row, col, prevFrame, row + i, col + j, frame);
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
    IplImage* prevFrame;
};

void calculate_candidate(void* args) {
    struct CandidateCalculateArgs* data = (struct CandidateCalculateArgs*) args;

    const vec_t dir = look_for_block(data->block->fila, data->block->columna, data->prevFrame, data->frame);
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
    IplImage *prevFrame = cvCloneImage(frame);
    IplImage *outputFrame = cvCloneImage(frame);

    const IplImage* primerFrame = cvCloneImage(frame);
    vec_t crop = (vec_t){0, 0};

    block_candidate_t candidates[NTHREADS] = {(block_candidate_t)
        { -1, -1, -1}};

    get_candidate_blocks(frame, candidates, NTHREADS);

#ifdef DEBUG
    for (int i = 0; i < NTHREADS; i++)
        printf("candidato %d: fila %d, columna %d, max_dif %d\n", i, candidates[i].fila, candidates[i].columna, candidates[i].max_dif);
#endif

    pthread_t threads[NTHREADS];
    struct CandidateCalculateArgs args[NTHREADS];

    while ((frame = cvQueryFrame(capture)) != NULL) {

        for (int i = 0; i < NTHREADS; i++) {
            args[i] = (struct CandidateCalculateArgs){
                &candidates[i],
                crop.i,
                crop.j,
                frame,
                prevFrame
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

        for (int i = 0; i < NTHREADS; i++) {
            args[i].block->fila += sum_cropY / NTHREADS;
            args[i].block->columna += sum_cropX / NTHREADS;
        }

#ifdef DEBUG
        printf("cropY: %d, cropX: %d\n", crop.i, crop.j);
#endif

        if (argc != 3) {
            crop_image(outputFrame, crop.i < 0 ? -crop.i : 0, crop.i > 0 ? crop.i : 0, crop.j < 0 ? -crop.j : 0, crop.j > 0 ? crop.j : 0);
#ifdef DEBUG
            for (int i = 0; i < NTHREADS; i++) {
                cvRectangle(prevFrame, cvPoint(candidates[i].columna, candidates[i].fila), cvPoint(candidates[i].columna + BLOCK_SIZE, candidates[i].fila + BLOCK_SIZE), cvScalar(0, 0, 255, 0), 1, LINE_MAX, 0);
            }
#endif
            cvShowImage("Prev", outputFrame);
            cvShowImage("Frame", prevFrame);
            cvWaitKey(0);
        }
        cvCopy(frame, prevFrame, NULL);
        cvCopy(primerFrame, outputFrame, NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &finish);
    elapsed = finish.tv_sec - start.tv_sec;
    elapsed += (finish.tv_nsec - start.tv_nsec) / 1000000000.0;
    printf("Tiempo de ejecución: %f\n", elapsed);
}
