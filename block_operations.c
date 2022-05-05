#include <stdio.h>
#include <stdlib.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#ifdef AVX2
#include <immintrin.h>
#else
#include <emmintrin.h>
#endif
#include "block_operations.h"

#ifdef AVX2

int block_compare(int i, int j, IplImage *imgA, int k, int l, IplImage* imgB) {
    int fila, columna;
    __m256i d = _mm256_set1_epi8(0);
    for (fila = 0; fila < BLOCK_SIZE; fila++) {
        __m256i *pImgA = (__m256i *) (imgA->imageData + (i + fila) * imgA->widthStep + j * imgA->nChannels);
        __m256i *pImgB = (__m256i *) (imgB->imageData + (k + fila) * imgB->widthStep + l * imgB->nChannels);
        for (columna = 0; columna < BLOCK_SIZE * imgA->nChannels; columna += 32) {
            __m256i a = _mm256_loadu_si256(pImgA++);
            __m256i b = _mm256_loadu_si256(pImgB++);
            __m256i c = _mm256_sad_epu8(a, b);
            d = _mm256_add_epi32(d, c);
        }
    }
    __m256i e = _mm256_add_epi32(_mm256_srli_si256(d, 8), d);
    return _mm256_cvtsi256_si32(e);
}

int block_intensity(IplImage* img, int i, int j) {
    const static unsigned char mask_arr[16] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0};
    const __m256i mask = _mm256_loadu_si256((__m256i*) mask_arr);
    __m256i suma = _mm256_set1_epi8(0);
    for (int fila = 0; fila < BLOCK_SIZE; fila++) {
        __m256i* pImg = (__m256i*) (img->imageData + (fila + i) * img->widthStep + j * img->nChannels);
        for (int columna = 0; columna < BLOCK_SIZE * img->nChannels; columna += 32) {
            __m256i i = _mm256_loadu_si256(pImg++);
            __m256i a = _mm256_sad_epu8(i, _mm256_set1_epi8(0));
            __m256i b = _mm256_srli_si256(a, 8);
            __m256i c = _mm256_and_si256(a, mask);
            suma = _mm256_add_epi32(suma, b);
            suma = _mm256_add_epi32(suma, c);
        }
    }
    return _mm256_cvtsi256_si32(suma);
}
#else

int block_compare(int i, int j, IplImage *imgA, int k, int l, IplImage* imgB) {
    int fila, columna;
    __m128i d = _mm_set1_epi32(0);
    for (fila = 0; fila < BLOCK_SIZE; fila++) {
        __m128i *pImgA = (__m128i *) (imgA->imageData + (i + fila) * imgA->widthStep + j * imgA->nChannels);
        __m128i *pImgB = (__m128i *) (imgB->imageData + (k + fila) * imgB->widthStep + l * imgB->nChannels);
        for (columna = 0; columna < BLOCK_SIZE * imgA->nChannels; columna += 16) {
            __m128i a = _mm_loadu_si128(pImgA++);
            __m128i b = _mm_loadu_si128(pImgB++);
            __m128i c = _mm_sad_epu8(a, b);
            d = _mm_add_epi32(d, c);
        }
    }
    __m128i e = _mm_add_epi32(_mm_srli_si128(d, 8), d);
    return _mm_cvtsi128_si32(e);
}

int block_intensity(IplImage* img, int i, int j) {
    const static unsigned char mask_arr[16] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0, 0, 0, 0, 0};
    const __m128i mask = _mm_loadu_si128((__m128i*) mask_arr);
    __m128i suma = _mm_set1_epi32(0);
    for (int fila = 0; fila < BLOCK_SIZE; fila++) {
        __m128i* pImg = (__m128i*) (img->imageData + (fila + i) * img->widthStep + j * img->nChannels);
        for (int columna = 0; columna < BLOCK_SIZE * img->nChannels; columna += 16) {
            __m128i i = _mm_loadu_si128(pImg++);
            __m128i a = _mm_sad_epu8(i, _mm_set1_epi8(0));
            __m128i b = _mm_srli_si128(a, 8);
            __m128i c = _mm_and_si128(a, mask);
            suma = _mm_add_epi32(suma, b);
            suma = _mm_add_epi32(suma, c);
        }
    }
    int x = _mm_cvtsi128_si32(suma);
    return x;
}
#endif