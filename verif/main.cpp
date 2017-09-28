#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include "verif_pe.h"
#include "sigint.h"
#include "pe_array.h"



int main(int argc, char **argv)
{
    int num_rows = 128;
    int num_cols = 64;

    /* in and out signals */
    VPeSignals vpes[num_rows];
    PeNSSignals ns[num_cols];

    /* structural components - tester and pe_array */
    ProcessingElementArray pe_array = ProcessingElementArray();
    VPeTester pe_tester(num_rows);

    size_t vpe_sz = sizeof(vpes[0].ew.weight.uint8);
    int rnd=open("/dev/urandom", O_RDONLY);

    (void)argc;
    (void)argv;


    /* connect pe array to pe tester */
    pe_tester.connect(&pe_array);

    /* initialize inputs to zero UINT8s */
    for (int i = 0; i < num_rows; i++) {
        vpes[i].clamp = 0;
        vpes[i].ew.pixel.raw = 0;
        vpes[i].ew.pixel_valid = 0;
        vpes[i].ew.weight.raw = 0;
        vpes[i].ew.weight_dtype = UINT8;
        vpes[i].ew.weight_toggle = 0;
    }

    /* clock 1 - pass in a weight and clamp*/
    vpes[0].ew.weight.uint8 = 23;
    vpes[0].clamp = true;
    pe_tester.write(vpes);
    pe_tester.step(); 

    /* clock 2 */
    /* stop weight feeding */
    vpes[0].clamp = false;
    vpes[0].ew.weight.uint8 = 0;
    /* toggle weight for use */
    vpes[0].ew.weight_toggle = true;
    /* setup first pixel */
    read(rnd, &vpes[0].ew.pixel.uint8, vpe_sz);
    vpes[0].ew.pixel_valid = true;
    /* step */
    pe_tester.write(vpes);
    pe_tester.step(); 
    /* clean up state, turn off weight toggle */
    vpes[0].ew.weight_toggle = false;

    /* clock 3 - N, send pixels */
    for (int i = 0; i < 130; i++) {
        read(rnd, &vpes[0].ew.pixel.uint8, vpe_sz);
        pe_tester.write(vpes);
        pe_tester.step(); 
        pe_tester.read(ns);
        printf("%d\n", ns[0].partial_sum.uint32);
    }
    close(rnd);
    return 0;

}

