
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

vec_t look_for_top(const int kBlockRow, const int kBlockCol, IplImage* prevFrame, IplImage* frame) {
    int i, j;

    for (i = -OFFSET; i < 0; i++) {
        for (j = -OFFSET; j < OFFSET; j++) {
            if (block_compare(kBlockRow, kBlockCol, prevFrame, kBlockRow + i, kBlockCol + j, frame) == 0) {
                return (vec_t){ i, j};
            }
        }
    }

    for (i = 1; i < OFFSET; i++) {
        for (j = -OFFSET; j < OFFSET; j++) {
            if (block_compare(kBlockRow, kBlockCol, prevFrame, kBlockRow + i, kBlockCol + j, frame) == 0) {
                return (vec_t){i, j};
            }
        }
    }
    return (vec_t){0, 0};
}

vec_t look_for_sides(const int kBlockRow, const int kBlockCol, IplImage* prevFrame, IplImage* frame) {
    int j;
    for (j = -1; j >= -OFFSET; j--) {
        if (block_compare(kBlockRow, kBlockCol, prevFrame, kBlockRow, kBlockCol + j, frame) == 0) {
            return (vec_t){0, j};
        }
    }
    for (j = 1; j <= OFFSET; j++) {
        if (block_compare(kBlockRow, kBlockCol, prevFrame, kBlockRow, kBlockCol + j, frame) == 0) {
            return (vec_t){0, j};
        }
    }
    return (vec_t){0, 0};
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
    const vec_t top = look_for_top(data->block->fila + data->cropY, data->block->columna + data->cropX, data->prevFrame, data->frame);
    const vec_t side = look_for_sides(data->block->fila + data->cropY, data->block->columna + data->cropX, data->prevFrame, data->frame);

    data->cropY = top.i;
    data->cropX = top.j;
    data->cropY += side.i;
    data->cropX += side.j;
}

int main(int argc, char** argv) {


    if (argc != 2 && argc != 3) {
        printf("Argumentos incorrectos\nestabilicacionvideo.exe [ruta archivo de vídeo] [-showoff]\n");
        exit(-1);
    }

    if (argc != 3 && strcmp(argv[2], "-showoff") != 0) {
        printf("Segundo argumento no reconocido\n");
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
    int cropY = 0, cropX = 0;

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
                cropY,
                cropX,
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

        cropY += sum_cropY / NTHREADS;
        cropX += sum_cropX / NTHREADS;

#ifdef DEBUG
        printf("cropY: %d, cropX: %d\n", cropY, cropX);
#endif

        if (argc != 3) {
            crop_image(outputFrame, cropY < 0 ? -cropY : 0, cropY > 0 ? cropY : 0, cropX < 0 ? -cropX : 0, cropX > 0 ? cropX : 0);
            cvShowImage("Prev", outputFrame);
            cvShowImage("Frame", frame);
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
