#include "img_operations.h"
#include <stdio.h>
#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "block_operations.h"

#ifdef AVX2
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif

/**
 * Recorta una imagen poniendo marcos negros en los bordes
 * @param img
 * @param top Tamaño en píxeles que tendrá el borde superior
 * @param bot Tamaño en píxeles que tendrá el borde inferior
 * @param left Tamaño en píxeles que tendrá el borde izquierdo
 * @param right Tamaño en píxeles que tendrá el borde derecho
 */
#ifdef AVX2

void crop_image(IplImage *img, int top, int bot, int left, int right) {
    const __m256i CEROS = _mm256_set1_epi8(0);
    int fila, columna;
    // Borde superior
    for (fila = 0; fila < top; fila++) {
        __m256i* pImg = (__m256i*) (img->imageData + fila * img->widthStep); // Se asignan con SSE los píxeles posibles
        for (columna = 0; columna + 32 < img->widthStep; columna += 32)
            _mm256_storeu_si256(pImg++, CEROS);

        // En caso de que el ancho de la imagen no sea múltiplo de 16 se termina de asignar los ceros uno a uno
        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < img->widthStep; columna++) {
            *pImgChar++ = 0;
        }
    }
    // Borde inferior

    for (fila = img->height - 1; fila > img->height - bot; fila--) {
        __m256i* pImg = (__m256i*) (img->imageData + fila * img->widthStep);
        for (columna = 0; columna + 32 < img->widthStep; columna += 32)
            _mm256_storeu_si256(pImg++, CEROS);

        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < img->widthStep; columna++)
            *pImgChar++ = 0;
    }

    // Borde izquierdo
    for (fila = 0; fila < img->height; fila++) {
        __m256i* pImg = (__m256i*) (img->imageData + fila * img->widthStep);
        for (columna = 0; columna + 32 < left * img->nChannels; columna += 32) {
            _mm256_storeu_si256(pImg++, CEROS);
        }
        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < left * img->nChannels; columna++)
            *pImgChar++ = 0;
    }
    // Borde derecho
    for (fila = 0; fila < img->height; fila++) {
        __m256i* pImg = (__m256i*) (img->imageData + fila * img->widthStep + (img->width - right) * img->nChannels);
        for (columna = (img->width - right) * img->nChannels; columna + 32 < img->widthStep; columna += 32)
            _mm256_storeu_si256(pImg++, CEROS);
        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < img->widthStep; columna++)
            *pImgChar++ = 0;
    }
}
#else

void crop_image(IplImage *img, int top, int bot, int left, int right) {
    const __m128i CEROS = _mm_set1_epi8(0);
    int fila, columna;
    // Borde superior
    for (fila = 0; fila < top; fila++) {
        __m128i* pImg = (__m128i*) (img->imageData + fila * img->widthStep); // Se asignan con SSE los píxeles posibles
        for (columna = 0; columna + 16 < img->widthStep; columna += 16) {
            _mm_storeu_si128(pImg++, CEROS);
        }
        // En caso de que el ancho de la imagen no sea múltiplo de 16 se termina de asignar los ceros uno a uno
        //unsigned char* pImgChar = (unsigned char*) (img->imageData + fila * img->widthStep + columna * img->nChannels);
        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < img->widthStep; columna++) {
            *pImgChar++ = 0;
        }
    }
    // Borde inferior

    for (fila = img->height - 1; fila > img->height - bot; fila--) {
        __m128i* pImg = (__m128i*) (img->imageData + fila * img->widthStep);
        for (columna = 0; columna + 16 < img->widthStep; columna += 16)
            _mm_storeu_si128(pImg++, CEROS);

        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < img->widthStep; columna++)
            *pImgChar++ = 0;
    }

    // Borde izquierdo
    for (fila = 0; fila < img->height; fila++) {
        __m128i* pImg = (__m128i*) (img->imageData + fila * img->widthStep);
        for (columna = 0; columna + 16 < left * img->nChannels; columna += 16) {
            _mm_storeu_si128(pImg++, CEROS);
        }
        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < left * img->nChannels; columna++)
            *pImgChar++ = 0;
    }
    // Borde derecho
    for (fila = 0; fila < img->height; fila++) {
        __m128i* pImg = (__m128i*) (img->imageData + fila * img->widthStep + (img->width - right) * img->nChannels);
        for (columna = (img->width - right) * img->nChannels; columna + 16 < img->widthStep; columna += 16)
            _mm_storeu_si128(pImg++, CEROS);
        unsigned char* pImgChar = (unsigned char*) (pImg);
        for (; columna < img->widthStep; columna++)
            *pImgChar++ = 0;
    }
}
#endif

void add_candidate(block_candidate_t arr[], block_candidate_t candidate, int n) {
    int posMin = 0;
    for (int i = 1; i < n; i++) {
        if (arr[i].max_dif < arr[posMin].max_dif) {
            posMin = i;
        }
    }
    arr[posMin] = candidate;
}

void get_candidate_blocks(IplImage* Img, block_candidate_t candidates[], int n) {
    for (int i = 0; i < n; i++) {
        candidates[i] = (block_candidate_t){-1, -1, -1, NULL};
    }
    for (int fila = Img->height / 3; fila < Img->height - Img->height / 3; fila += BLOCK_SIZE * 1.5) {
        for (int columna = Img->width / 3; columna < Img->width - Img->width / 3; columna += BLOCK_SIZE * 1.5) {
            int intensidad_centro = block_intensity(Img, fila, columna);
            for (int fila2 = fila - BLOCK_SIZE; fila2 < fila + BLOCK_SIZE; fila2 += BLOCK_SIZE) {
                for (int columna2 = columna - BLOCK_SIZE; columna2 < columna + BLOCK_SIZE; columna2 += BLOCK_SIZE) {
                    add_candidate(candidates, (block_candidate_t){fila2, columna2, abs(intensidad_centro - block_intensity(Img, fila2, columna2))}, n);
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
#ifdef DEBUG
        printf("candidato %d: fila %d, columna %d, max_dif %d\n", i, candidates[i].fila, candidates[i].columna, candidates[i].max_dif);
#endif
        candidates[i].block = cvCreateImage(cvSize(BLOCK_SIZE, BLOCK_SIZE), Img->depth, Img->nChannels);
        for (int fila = 0; fila < BLOCK_SIZE; fila++) {
#if AVX2
            __m256i* pImg = (__m256i*) (Img->imageData + (fila + candidates[i].fila) * Img->widthStep + candidates[i].columna * Img->nChannels);
            __m256i* pBlock = (__m256i*) (candidates[i].block->imageData + fila * candidates[i].block->widthStep);
            for (int columna = 0; columna < BLOCK_SIZE * Img->nChannels; columna += 32) {
                _mm256_storeu_si256(pBlock++, _mm256_loadu_si256(pImg++));
            }
#else
            __m128i* pImg = (__m128i*) (Img->imageData + (fila + candidates[i].fila) * Img->widthStep + candidates[i].columna * Img->nChannels);
            __m128i* pBlock = (__m128i*) (candidates[i].block->imageData + fila * candidates[i].block->widthStep);
            for (int columna = 0; columna < BLOCK_SIZE * Img->nChannels; columna += 16) {
                _mm_storeu_si128(pBlock++, _mm_loadu_si128(pImg++));
            }
#endif
        }
    }

}
