#pragma once

#include <iostream>
#include <immintrin.h>
#include <xmmintrin.h>
#include <vector>


typedef unsigned char uchar;
typedef unsigned short ushort;

using namespace std;

class RleRegion
{
public:
    RleRegion() {
        cb = nullptr;
        ce = nullptr;
        row = nullptr;
        size = 0;
        area = 0;
    }
    ~RleRegion() {
        if (cb) {
            delete[]cb;
            delete[]ce;
            delete[]row;
        }
    }
    static std::shared_ptr<RleRegion> threshold_rle(uchar* data, int width, int height, int low, int high) {

        RleRegion* r = new RleRegion;

        int maxWidth = width / 2 + 1;

        ushort* buffer = new ushort[2 * maxWidth * height];

        int* Counts = new int[height];
        memset(Counts, 0, sizeof(int) * height);
        

#pragma omp parallel for
        for (int i = 0; i < height; i++) {

            uchar* rowPtr = data + i * width;
            ushort* bufferRow = buffer + i * 2 * maxWidth;

            int currentPreValue = 0;
            int currentCount = Counts[i];
            for (int j = 0; j < width; j += 32) {

                __m256i v = _mm256_load_si256((__m256i*)(rowPtr + j));
                __m256i v2 = _mm256_min_epu8(_mm256_max_epu8(v, _mm256_set1_epi8(low)), _mm256_set1_epi8(high));
                __m256i v3 = _mm256_cmpeq_epi8(v2, v);
                unsigned int mask = _mm256_movemask_epi8(v3);

                int bitOffset = 0;
                bool isContinue = false;
                while (mask) {
                    unsigned int skip = _tzcnt_u32(mask);
                    unsigned int skip2 = _tzcnt_u32(~mask);

                    currentPreValue += skip2;
                    if (currentPreValue && skip != 0) {
                        bufferRow[currentCount * 2] = j + bitOffset - 1;
                        bufferRow[currentCount * 2 + 1] = currentPreValue;
                        currentPreValue = 0;
                        currentCount++;
                    }
                    mask >>= (skip + skip2);
                    bitOffset += (skip + skip2);
                    if (bitOffset == 32) {
                        isContinue = (skip2 == 0);
                        break;
                    }
                }
                if (currentPreValue && (isContinue || (~mask))) {
                    bufferRow[currentCount * 2] = j + bitOffset - 1;
                    bufferRow[currentCount * 2 + 1] = currentPreValue;
                    currentPreValue = 0;
                    currentCount++;
                }
            }
            if (currentPreValue) {
                bufferRow[currentCount * 2] = width - 1;
                bufferRow[currentCount * 2 + 1] = currentPreValue;
                currentCount++;
            }
            Counts[i] = currentCount;
        }


        int totalCounts = 0;
        for (int i = 0; i < height; i++) {
            totalCounts += Counts[i];
        }


        r->cb = new ushort[totalCounts];
        r->ce = new ushort[totalCounts];
        r->row = new ushort[totalCounts];

        int TotalCvx = 0;
        for (int i = 0; i < height; i++) {
            int SubCount = Counts[i];
            ushort* bufferRow = buffer + i * 2 * maxWidth;
            for (int j = 0; j < SubCount; j++) {
                int SubCvx = bufferRow[j * 2 + 1];
                r->cb[TotalCvx] = bufferRow[j * 2] - SubCvx + 1;
                r->ce[TotalCvx] = bufferRow[j * 2];
                r->row[TotalCvx] = i;
                r->area += SubCvx;

                TotalCvx++;
            }
        }
        r->size = TotalCvx;

        delete[]buffer;
        delete[]Counts;

        return std::shared_ptr<RleRegion>(r);
    }

    std::vector<std::shared_ptr<RleRegion>> connection() {


    }
    int Area() {

        return area;
    }

public:
    ushort* cb;
    ushort* ce;
    ushort* row;
    int size;
    int area = 0;
};
